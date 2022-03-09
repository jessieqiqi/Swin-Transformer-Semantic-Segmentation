import argparse
import numpy as np
from ast import parse
from os import makedirs
from os.path import basename, splitext
from PIL import Image

from mmseg.apis import init_segmentor, inference_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette


configs = [
    'configs/swin/upernet_swin_base_patch4_window7_512x512_160k_ade20k.py'
]
checkpoints = [
    'checkpoints/upernet_swin_base_patch4_window7_512x512.pth'
]

device = 'cuda:0'


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint_index", type=int, help="index of checkpoint")
    parser.add_argument("-i", "--img_path", type=str, help="path of an image which the model inference")
    parser.add_argument("-i_origin", "--original_img_path", type=str, help="path of the original image with overlapping bounding box.The scale ratio must be equal to the input image.")
    parser.add_argument("-l", "--list", action="store_true", help="list information about available checkpoints")
    args = parser.parse_args()

    if args.list:
        print("\nList available checkpoints:")
        for i, c in enumerate(checkpoints):
            print(f"index: {i} | path: {basename(c)}")
        print("\n")

    if args.checkpoint_index is None or args.img_path is None:
        return

    if len(checkpoints) < args.checkpoint_index:
        print(f"There are only {len(checkpoints)} available chechkpoints. Please confirm available checkpoints to use -c or --checkpoint option")
        return

    return args


def save_masked_image(img_path, result):
    makedirs(f'demo/dst_{splitext(basename(img_path))[0]}', exist_ok=True)
    img = np.array(Image.open(img_path))
    result = np.array(result)[0]
    
    # ADE20K has 150 classes
    for i in [12, 24, 67]:
        mask = result == np.full(result.shape, i)
        mask = mask.astype(np.uint8)
        if mask.shape[:2] != img.shape[:2]:
            mask = np.array(Image.fromarray(mask).resize((img.shape[1], img.shape[0]), Image.BICUBIC))
        mask = mask.reshape(*mask.shape, 1)

        dst = img * mask
        dst_path = f'demo/dst_{splitext(basename(img_path))[0]}/{splitext(basename(img_path))[0]}_class{str(i).zfill(3)}.jpg'
        Image.fromarray(dst.astype(np.uint8)).save(dst_path)


def demo(img_path, config_file, checkpoint_file):
    model = init_segmentor(config_file, checkpoint_file, device=device)
    result = inference_segmentor(model, img_path)
    show_result_pyplot(model, img_path, result, get_palette('ade20k'))

    # save result
    output_path = f'demo/{splitext(basename(img_path))[0]}_output.jpg'
    model.show_result(img_path, result, palette=get_palette('ade20k'), out_file=output_path)
    return result


def main():
    args = parser()
    if args is None:
        return args

    checkpoint_index  = args.checkpoint_index
    img_path          = args.img_path
    original_img_path = args.original_img_path if args.original_img_path else args.img_path
    
    result = demo(img_path, configs[checkpoint_index], checkpoints[checkpoint_index])
    save_masked_image(original_img_path, result)


if __name__ == "__main__":
    main()