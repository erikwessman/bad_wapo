import os
import cv2
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import argparse
import numpy as np


SYMBOL_MAP = {
    255: 'I',
    223: 'J',
    183: 'L',
    177: 'T',
    167: 'Y',
    161: 'F',
    160: 'C',
    148: 'V',
    147: 'E',
    144: 'P',
    137: 'S',
    131: 'Z',
    128: 'X',
    117: 'A',
    114: 'U',
    110: 'K',
    108: 'G',
    107: 'R',
    81: 'H',
    71: 'O',
    69: 'D',
    60: 'B',
    37: 'N',
    34: 'Q',
    0: 'W',
}


def main(args):
    cap = cv2.VideoCapture(args.path)
    ret, frame = cap.read()

    if not ret:
        print("Unable to read video")
        exit(1)

    out = cv2.VideoWriter(
        os.path.join("./", "out.avi"),
        cv2.VideoWriter_fourcc(*"XVID"),
        30.0,
        (frame.shape[1], frame.shape[0]),
    )

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    font_path = './ARIALBD.ttf'
    font_size = 20
    font = ImageFont.truetype(font_path, font_size)

    with tqdm(total=total_frames, desc="Processing frames") as progress:
        while ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 

            pixel_counter = 1
            block_size = 40
            height, width = frame.shape

            pil_image = Image.new('RGB', (width, height), 'white')
            draw = ImageDraw.Draw(pil_image)

            for y in range(0, height, block_size):
                for x in range(0, width, block_size):
                    block = frame[y:y+block_size, x:x+block_size]
                    average_intensity = np.mean(block)

                    closest_key = min(SYMBOL_MAP.keys(), key=lambda k: abs(k - average_intensity))
                    pixel_symbol = SYMBOL_MAP[closest_key]

                    center_x = x + block_size // 2
                    center_y = y + block_size // 2
                    draw.text((center_x - 8, center_y - 8), pixel_symbol, fill='black', font=font)
                    draw.text((x, y), str(pixel_counter), fill='black')

                    pixel_counter += 1

            for x in range(0, width, block_size):
                draw.line([(x, 0), (x, height)], fill="gray", width=1)

            for y in range(0, height, block_size):
                draw.line([(0, y), (width, y)], fill="gray", width=1)

            output_frame = np.array(pil_image)
            output_frame = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)

            cv2.imshow("Output", output_frame)
            out.write(output_frame)

            ret, frame = cap.read()

            progress.update(1)

            if cv2.waitKey(30) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--path", required=True, help="")
    args = parser.parse_args()

    main(args)    