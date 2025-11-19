import h5py
import numpy as np
import cv2

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("target_hdf5_path", type=str, help="HDF5 file to be visualized")
parser.add_argument("--demo_idx", type=int, default=0, help="Index of demonstration")
parser.add_argument("--view_idx", type=int, default=1, choices=[1, 2], help="Index of camera view")
#parser.add_argument("--video_width", type=int, default=480, help="Width of video frames")
#parser.add_argument("--video_height", type=int, default=640, help="Height of video frames")
parser.add_argument("--video_fps", type=int, default=10, help="Frame per sec. of video")
args = parser.parse_args()


with h5py.File(args.target_hdf5_path, "r") as f:
    images = np.array(f[f"data/demo_{args.demo_idx}/obs/front_image_{args.view_idx}"])
    actions = np.array(f[f"data/demo_{args.demo_idx}/actions"])

height, width = images.shape[1:3]
#print(f"[DEBUG] (height, width) = ({height}, {width})")
#raise AssertionError

# Reshape the actions to [T, 10, 4]
T = len(actions)
actions = actions.reshape((T, 10, 4))


if args.view_idx == 1:
    actions = actions[:, :, :2]
else:  # args.view_idx == 2
    actions = actions[:, :, 2:]


fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video = cv2.VideoWriter("output.mp4", fourcc, args.video_fps, (width, height))

for i in range(images.shape[0]):
    img = cv2.cvtColor(images[i].copy(), cv2.COLOR_RGB2BGR)
    action = actions[i]

    action_unscaled = action * np.array([width, height])

    for i, pt in enumerate(action_unscaled):
        g = int(255 - (255 - 100) * (i / (len(action_unscaled) - 1)))
        img = cv2.circle(img, (int(pt[1]), int(pt[0])), radius=5, color=(0, g, 0), thickness=-1)

    video.write(img)

video.release()

print("The video has been successfully saved as output.mp4")

