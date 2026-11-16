import cv2
from pathlib import Path
from typing import Union, List, Tuple

def create_project(folder: Union[str, Path]) -> None:
    """
    Create a project for social seeking quantification
    Creates the folder structure
    Params:
        folder (str, Path): path to a folder to create the project into
    Returns:
        None
    """
    folder.mkdir() # create project root
    folders_to_create = ['tracks', 'videos', 'classifiers', 'trackers', 'labels', 'predictions', 'logs']
    for folder_name in folders_to_create:
        (folder / folder_name).mkdir()
    
def get_cage_coordinates(video: str) -> Union[list, None]:
    """
    Get cage coordinates for each cage in a raw video
    Parameters:
        video (str): Path to the video file.
    Returns:
        None
    """
    crop_regions: List[Tuple[int, int]] = []
    current_points: List[Tuple[int, int]] = []
    drawing = False
    frame_copy = None

    def draw_rectangle(event, x: int, y: int, flags, param):
        nonlocal current_points, drawing, frame_copy

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            if x < 0:
                x = 0
            if y < 0:
                y= 0
            current_points = [(x, y)] # Store coordinates of point when mouse clicked
            frame_copy = frame.copy()  # Store a copy of the original frame

        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            img = frame_copy
            cv2.rectangle(img, current_points[0], (x, y), (0, 255, 255), thickness=2) # show the rectangle as you're drawing it.*
            cv2.imshow('Crop', img)

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            current_points.append((x, y)) # Store the points where the mouse stopped clicking
            crop_regions.append(tuple(current_points))
            cv2.rectangle(frame, current_points[0], current_points[1], (0, 255, 255), 2)
            cv2.imshow('Crop', frame)
            current_points = []

    cap = cv2.VideoCapture(video)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    ret, frame = cap.read()
    original = frame.copy()
    cv2.namedWindow('Crop')
    cv2.setMouseCallback('Crop', draw_rectangle)

    while True:
        cv2.imshow('Crop', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):

            cv2.destroyAllWindows()
            cap.release()
            return
        elif key == ord('c'):
            break
        elif key == ord('x'):
            if crop_regions:
                crop_regions.pop() # discard last rectangle
            frame = original.copy()
            for top_left, bottom_right in crop_regions:
                cv2.rectangle(frame, top_left, bottom_right, (0, 255, 255), 2)
            current_points = []
            drawing = False
            cv2.imshow('Crop', frame)

    crop_regions = clip_coords(crop_regions,
                               width,
                               height)
    return crop_regions # coordinates of the crop in [((x1, y1), (x2, y2), (x1, y1), (x2, y2))] format

def clip_coords(crop_coords: List[Tuple[int, int]],
                width: int,
                height: int) -> List[Tuple[int, int]]:
    """
    Clip coordinates to stay above 0 and below width or height
    Params:
        crop_coords (Tuple[int, int]): pair of cropping coordinates
    Returns:
        clipped_coords (Tuple[int, int]): pair of clipped cropping coordinates
    """
    clipped_coords = []
    for video in crop_coords:
        video_clipped_coords = []
        for point in video:
            if point[0] > width:
                new_point_x = int(width)
            elif point[0] < 0:
                new_point_x = 0
            else:
                new_point_x = point[0]
            if point[1] > height:
                new_point_y = int(height)
            elif point[1] < 0:
                new_point_y = 0
            else:
                new_point_y = point[1]
            video_clipped_coords.append((new_point_x, new_point_y))
        video_clipped_tuple = tuple(video_clipped_coords)
        clipped_coords.append(video_clipped_tuple)

    return clipped_coords


def write_crop_coords_for_camera(outfile: str, videos: List[str]) -> None:
    """
    Write crop coords of individual cages for each camera in the setup into a .csv file
    Params:
        outfile (str): path of the .csv file to be saved
        videos (list(str)): list of paths to video to extract coordinates from.
        videos should contain one video per camera
    Returns:
        None
    """
    with open(outfile, 'w', encoding='utf-8') as f:
        f.write("index,x1,y1,x2,y2\n")
        for video in videos:
            video_name = Path(video).stem
            num_camera = video_name[0]
            crop_coord = get_cage_coordinates(video)
            if crop_coord is None:
                return
            for i, cage in enumerate(crop_coord):
                f.write(f'{num_camera}_{i+1},{cage[0][0]},{cage[0][1]},{cage[1][0]},{cage[1][1]}\n')


