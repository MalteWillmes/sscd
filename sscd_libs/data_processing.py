# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 19:45:28 2021

@author: Bruno Caneco

Module for utility functions dealing with data/image preparation and processing

"""

# import standard libraries
import os
from pathlib import Path
import collections
import hashlib
import concurrent.futures
import logging
import math

# import installed packages/libraries
from PIL import Image
from tqdm import tqdm
import numpy as np
from xml.etree import ElementTree
import pandas as pd


# import local modules
from tools.diagonal_crop import crop


logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# accepted input image types (matched case-insensitively)
IMAGE_EXTENSIONS = (".tif", ".tiff", ".jpg", ".jpeg")


def to_8bit_rgb(im):
    """
    Convert a PIL image of any common mode to 8-bit RGB, the input the detectors
    were trained on. 16-bit (and other high bit-depth) greyscale is scaled down
    to 8 bits rather than clipped; alpha channels are dropped.
    """
    if im.mode in ("I;16", "I;16B", "I;16L", "I;16N", "I", "F"):
        pixels = np.asarray(im, dtype=np.float64)
        # 16-bit images use the full 0-65535 range; beyond that, stretch to the data range
        if pixels.max() <= 255:
            scaled = pixels
        elif pixels.max() <= 65535 and pixels.min() >= 0:
            scaled = pixels / 257
        else:
            lo, hi = pixels.min(), pixels.max()
            scaled = (pixels - lo) / (hi - lo) * 255
        im = Image.fromarray(np.clip(np.round(scaled), 0, 255).astype(np.uint8), mode="L")
    return im if im.mode == "RGB" else im.convert("RGB")


def tiff_to_jpg(tiff_input_filepath, jpg_output_filepath):
    """
    Converts a tiff (or jpeg) image to an 8-bit RGB jpeg.

    Args
    ----------
    tiff_input_filepath : str
        filepath to tiff image file to be converted.
    jpg_output_filepath : str
        filepath to write the jpeg image.
    
    """
    
    with Image.open(tiff_input_filepath) as im:
        if getattr(im, "n_frames", 1) > 1:
            logger.warning("%s has %d pages/frames - only the first is used",
                           Path(tiff_input_filepath).name, im.n_frames)
        to_8bit_rgb(im).save(jpg_output_filepath, 'JPEG', quality=95)
    
    
    
    
# ------------------------------------------------------------------------------
def list_input_images(input_imgs_dir):
    """
    Scale image files in a directory (.tif/.tiff/.jpg/.jpeg, any case), sorted.

    Raises FileNotFoundError if there are none, and ValueError if two share a
    file name (ignoring extension), as they would overwrite each other's results.
    """
    # extension matched case-insensitively, without glob, so directory names
    # containing [ ] etc. work too
    img_input_fpaths = sorted(
        str(p) for p in Path(input_imgs_dir).iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
    if len(img_input_fpaths) == 0:
        raise FileNotFoundError(f"No images of type {', '.join(IMAGE_EXTENSIONS)} found in {input_imgs_dir}")

    stems = collections.Counter(Path(p).stem.lower() for p in img_input_fpaths)
    duplicates = sorted(stem for stem, n in stems.items() if n > 1)
    if duplicates:
        raise ValueError("Several input images share the same file name (ignoring extension): "
                         + ", ".join(duplicates))
    return img_input_fpaths


# ------------------------------------------------------------------------------
def find_duplicate_images(img_filepaths):
    """
    Groups of image files with identical content (e.g. 'scale.tif' and 'scale (1).tif'
    from copying a file twice), which would otherwise be processed - and counted - twice.

    Cheap even for large files on a network share: only files of the same size are
    compared, first by their start and end, and only remaining candidates in full.
    Returns a list of groups (lists of paths), each with 2+ identical files.
    """
    def digest(path, full):
        h = hashlib.blake2b(digest_size=16)
        with open(path, "rb") as f:
            if full:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            else:
                h.update(f.read(1 << 18))
                f.seek(max(os.path.getsize(path) - (1 << 18), 0))
                h.update(f.read(1 << 18))
        return h.hexdigest()

    def group_by(paths, key):
        groups = collections.defaultdict(list)
        for p in paths:
            groups[key(p)].append(p)
        return [g for g in groups.values() if len(g) > 1]

    duplicates = []
    for same_size in group_by(img_filepaths, os.path.getsize):
        for same_ends in group_by(same_size, lambda p: digest(p, full=False)):
            duplicates.extend(group_by(same_ends, lambda p: digest(p, full=True)))
    return [sorted(g) for g in duplicates]


# ------------------------------------------------------------------------------
def images_tiff_to_jpeg(input_imgs_dir, output_imgs_dir, progress=None):
    
    """
    Wrapper to convert tiff images located within the directory into jpeg format.
    
    Args
    -----
    input_imgs_dir : str
        path to directory where image files are located. Input formats accepted: 
            TIFF/TIF and JPG/JPEG (any case). All are re-encoded as 8-bit RGB jpegs
            in the output directory
    output_imgs_dir : str
        path directory where jpeg image files should be written
    progress : callable, optional
        called as progress(done, total) after each converted image
        
        
    Returns
    -------
    0 to indicate successful completion
    
    """

    img_input_fpaths = list_input_images(input_imgs_dir)
        
    # --- generate output filepaths for converted images
    img_output_fpaths = [os.path.join(output_imgs_dir, Path(name).stem + '.jpg') 
                         for name in img_input_fpaths]
    
    # convert images in parallel with a few threads (Pillow releases the GIL
    # while decoding/encoding). A process pool re-imports sscd.py - and with it
    # TensorFlow - in every worker on Windows, multiplying memory use.
    max_workers = min(4, os.cpu_count() or 1)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        
        logger.info("Converting %d images to jpeg format", len(img_input_fpaths))
        
        # use the executor to map the converting function to the iterable of input paths
        converted = executor.map(tiff_to_jpg, img_input_fpaths, img_output_fpaths)
        for i, _ in enumerate(tqdm(converted, total=len(img_input_fpaths), ascii=True, ncols=120), 1):
            if progress:
                progress(i, len(img_input_fpaths))
        
    #return 0



# ------------------------------------------------------------------------------
def pol2cart(radius, phi):
    x = radius * np.cos(phi)
    y = radius * np.sin(phi)
    
    return [x, y]


# ------------------------------------------------------------------------------
def get_transect_base_coords(focus_centre, transect_rad, transect_width):
     
    """
    Compute the coordinates of the transect's base point, defined as the upper left point of the 
    area to be cropped before rotation. 
    Approached using a polar-to-cartesian system conversion, using the focus centre as the pole, 
    as the base's angle and distance to the focus centre are always known
    
    Args
    ----
        focus_centre: tuple
            (x,y) coordinates of the focus
        transect_rad: float
            Angle of the transect relative to image's horizontal line, in radians     
        transect_width: float
            Width of the transect, in pixels
        
    Returns
    -------
        A tuple with the (x,y) coordinate of the transect's base point 
    
    """
    
    # Get polar coordinates
    # get angle (radians) of base point in the polar plane - always perpendicular to transect angle
    # Note: need to work with inverted angles as the y-axis for images is inverted 
    # (i.e. coord (0,0) is at the top left of image)
    base_rad = (2*math.pi - transect_rad) - math.pi/2
    # get radial distance of base point = distance to focus center, known to be half of the transect width
    base_radius = transect_width/2
    
    # Convert to cartesian coords (relative to the pole)
    base_xy_0 = pol2cart(base_radius, base_rad)
    
    # Project coords in the image's coordinate space
    base_xy = (focus_centre[0] + base_xy_0[0], focus_centre[1] + base_xy_0[1])
    
    return base_xy



# ------------------------------------------------------------------------------
def line_x_rectangle(a, b, x_min, y_min, x_max, y_max):
    
    """
    Find the intersection points between a continuos line and the edges of a rectangle 
    Based on line Clipping following Liang-Barsky Algorithm (source: https://twinee.fr/2020-03-24-Liang-Barsky/)
    
    Args
    ----
        a: float
            slope of the line
        b: float
            y-intersept of the line
        x_min, y_min, x_max, y_max : float
            rectangle edges
    
    Returns
    -------
        Nested tuple ((x1,y1), (x2,y2)), coordinates of the two intercection points 
        between the line and the rectangle
    
    """
    
    # breakpoint()

    # Intersections of f(x) = ax + b with the rectangle. (x, y, axis)
    p1, p2 = (x_min, a * x_min + b, 'x'), (x_max, a * x_max + b, 'x'), 
    p3, p4 = ((y_min - b) / a, y_min, 'y'), ((y_max - b) / a, y_max, 'y')
    # Python sorts them using the first key
    p1, p2, p3, p4 = sorted([p1, p2, p3, p4])
    
    # # Check if there is an intersection, returns the points otherwise
    # if p1[2] == p2[2]:
    #     return None
    # return p2[:2], p3[:2]
    #
    # BC: Commented out above chunk as I'n not sure if it's correct... 
    # If the line sits above or below the rectangle, p1[2] != p2[2] and so the 
    # condition doesn't hold.
    # Actually, the condition is excluding the special case when the line crosses
    # exactly the opposite corners of the rectangle. In that case p1 == p2 and 
    # p3 == p4, so returning p2 and p3 should give the desired output
    # Warning: only returning p2 and p3 doesn't account for the no intersection case,
    # but on the context of transects in scales, where the focus is always inside
    # the image, there is always an intersection
    
    return p2[:2], p3[:2]


# ------------------------------------------------------------------------------
def get_transect_length(focus_centre, transect_rad, im_width, im_height):
    
    """
    Determine the length of the transect based on focus location and angle
    
    Args
    ----
        focus_centre: tuple
            (x,y) coordinates of the focus
        transect_rad: float
            Angle of the transect relative to image's horizontal line, in radians
        im_width: int
            Image width, in pixels
        im_height: int
            Image height, in pixels
    
    Returns
    -------
        Integer, the length of the transect, in pixels
    
    """
   
    trans_slope = math.tan(2*math.pi - transect_rad) 
    trans_yint = (focus_centre[1] - trans_slope*focus_centre[0])
    
    # Getting the points of intersection between the transect line and the image limits
    # Using the Liang-Barsky Algorithm, with transect is interpreted as a linear function.
    # Thus, we get 2 intersection points on opposite sides of the image - sorted from left to right
    if trans_slope == 0:
        trans_x_img = ((0,trans_yint), (im_width,trans_yint))
    else:
        trans_x_img = line_x_rectangle(
        a = trans_slope, 
        b = trans_yint, 
        x_min = 0, 
        y_min = 0,
        x_max = im_width, 
        y_max = im_height
        )
    
    #breakpoint()
    
    # select which intersection to use 
    # 1st intersection if transect in 2nd or 3rd quadrants, 
    # 2nd intersection otherwise
    if 1/2*math.pi <= transect_rad < 3/2*math.pi:
        trans_int = trans_x_img[0]
        #trans_int = (trans_int[0], trans_int[1] + im_height)
    else:
        trans_int = trans_x_img[1]
        #trans_int = (trans_int[0], trans_int[1] - im_height)
    
    # compute transect length using pythagoras theorem
    trans_adj_lt = focus_centre[0] - trans_int[0]    
    trans_opp_lt = focus_centre[1] - trans_int[1]
    trans_lt = math.sqrt(trans_adj_lt**2 + trans_opp_lt**2)     
        
    return trans_lt
    



# ------------------------------------------------------------------------------
TransectGeometry = collections.namedtuple("TransectGeometry", ["base", "angle_rad", "width", "length"])


def transect_geometry(focus_bbox, angle_deg, im_width=None, im_height=None):
    """
    Geometry of the radial transect at `angle_deg` from a detected focus.

    Single source of truth for both cutting transects out of a scale image
    (get_transects) and mapping transect detections back onto it
    (transect_to_image), so the two can never drift apart.

    Args
    ----
        focus_bbox: dict-like with xmin, ymin, xmax, ymax of the focus (pixels)
        angle_deg: transect angle in degrees, counter-clockwise from the image's
            positive x axis (0 = right, 90 = up)
        im_width, im_height: scale image size in pixels (only needed for the length)

    Returns
    -------
        TransectGeometry(base, angle_rad, width, length): base is the (x, y)
        corner of the transect next to the focus (the transect image's top-left
        pixel), width is the transect's height across (pixels) and length its
        extent from the focus to the image border (pixels; None without image size).
    """
    xmin, ymin, xmax, ymax = (focus_bbox[k] for k in ("xmin", "ymin", "xmax", "ymax"))
    focus_centre = ((xmin + xmax)/2, (ymin + ymax)/2)
    width = min(xmax - xmin, ymax - ymin)/2
    angle_rad = math.radians(angle_deg)
    base = get_transect_base_coords(focus_centre, angle_rad, width)
    length = None
    if im_width is not None and im_height is not None:
        length = get_transect_length(focus_centre, angle_rad, im_width, im_height)
    return TransectGeometry(base, angle_rad, width, length)


def transect_to_image(geom, u, v):
    """
    Map transect-image pixel coordinates to scale-image coordinates.

    u runs along the transect (the transect image's x axis, 0 at the focus end)
    and v across it (the transect image's y axis). Works on scalars or arrays.

    Returns (x, y) in the scale image.
    """
    cos_a, sin_a = math.cos(geom.angle_rad), math.sin(geom.angle_rad)
    x = geom.base[0] + u*cos_a + v*sin_a
    y = geom.base[1] - u*sin_a + v*cos_a
    return x, y




# ------------------------------------------------------------------------------
def get_transects(focus_bbox, transect_degrees, img_filepath, output_dir):
    """
    Extract, and store, images of radial transects off the scale's focus
    
    Args:
    -----
        focus_bbox: dict
            focus bounding box limits (absolute measurements)
        
        transect_degrees: list
            Transect's radial angle(s) (in degrees)
        
        img_filepath: str
            path to scale image
            
        output_dir: str
            path to folder where transect images will be stored
        
    Returns:
    --------
        empty (saves images to output_dir)
        
    """
    
    im = Image.open(img_filepath)
    im.load()  # read pixels now so the file handle can be closed
    im_width, im_height = im.size
    img_id = Path(img_filepath).stem
    
    for angle_deg in transect_degrees:     
        
        geom = transect_geometry(focus_bbox, angle_deg, im_width, im_height)
        if geom.length < 1:
            # focus sits on the image border in this direction: nothing to crop
            logger.warning("Skipping transect %s_%s: focus lies on the image border", img_id, angle_deg)
            continue
        cropped_im = crop(im, geom.base, geom.angle_rad, geom.width, geom.length)
        transect_outfile = os.path.join(output_dir, img_id + f'_{angle_deg}.jpg')
        cropped_im.save(transect_outfile, 'JPEG')
         




# ------------------------------------------------------------------------------
def pascal_to_evaltxt(ann_dir, ann_id, out_dir):
    """
    Convert object bounding boxes annotation files in Pascal VOC format to text 
    files formatted in accordance with evaluator requirements

    TODO
    Parameters
    ----------
    ann_filepath : TYPE
        DESCRIPTION.
    out_dir : TYPE
        DESCRIPTION.

    Returns
    -------
    0 indicating successful completion

    """
    
    ann_filepath = os.path.join(ann_dir, ann_id + ".xml")
    
    # list of bounding box annotations to eventually write to txt
    bboxes = []
    
    # load the contents of the annotations file into an ElementTree
    tree = ElementTree.parse(ann_filepath)  # noqa: S314 - the user's own local annotation files
    
    for obj in tree.iter("object"):
        
        label = obj.find("name").text
        bndbox = obj.find("bndbox")
        bbox_min_x = int(float(bndbox.find("xmin").text))
        bbox_min_y = int(float(bndbox.find("ymin").text))
        bbox_max_x = int(float(bndbox.find("xmax").text))
        bbox_max_y = int(float(bndbox.find("ymax").text))
                
        # include this box in the list we'll return
        box = {
            "label": label,
            "xmin": bbox_min_x,
            "ymin": bbox_min_y,
            "xmax": bbox_max_x,
            "ymax": bbox_max_y,
            }
        
        bboxes.append(box)
        
    # pandas offers a convenient way to save dicT to txt files
    bboxes_df = pd.DataFrame(bboxes)
    # sort annotation boxes by xmin (i.e. from left to right)
    bboxes_df.sort_values(by=['xmin'], inplace = True, ignore_index=True)
    
    txt_filepath = os.path.join(out_dir, ann_id + ".txt")
    bboxes_df.to_csv(txt_filepath, header = False, sep = ' ', index=False)
    
    return 0




# ---------------------------------------------------------------------------------
def generate_label_map(labels_df: pd.DataFrame, output_dir, dict_filename, names_filename):
    """
    Generate a label map file of the classes of interest - i.e defines a mapping from string label names to 
    integer class IDs. Function writes out label maps in two different formats
            - as a dictionary encoded in a text file (.txt)
            - as a NAMES file (.names), simply comprising one label name per line
    
    BC: Stolen and hacked from https://github.com/monocongo/cvdata/blob/master/src/cvdata/convert.py

    Parameters
    ----------
    labels_df : pd.DataFrame
        DataFrame comprising label names, one per row.
    output_dir : str
        output directory to which files will be written to
    dict_filename : str
        name of file to comprise the dict-style label map.
    names_filename : str
        name of file to comprise the NAMES-style label names.

    Returns
    -------
    None.

    """
  
    # make the directory where the files will saved to, in case it doesn't yet exist
    os.makedirs(output_dir, exist_ok=True)

    # dictionary of labels to indices that we'll populate and write out
    label_indices = {}
    
    label_index = 1
    for label in labels_df["class"].unique():
        label_indices[label] = label_index
        label_index += 1
                
    # write out dictionary
    with open(os.path.join(output_dir, dict_filename), 'w') as f:
        print(label_indices, file=f)
        
    # write out as NAMES  file
    with open(os.path.join(output_dir, names_filename), 'w') as f:
        f.writelines(labels_df["class"].unique())
        
# ---------------------------------------------------------------------------------




