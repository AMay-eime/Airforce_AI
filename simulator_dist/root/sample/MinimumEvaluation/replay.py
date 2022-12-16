import os
from argparse import ArgumentParser
from ASRCAISim1.viewer.GodViewLoader import GodViewLoader


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--movie-dir', default='./log')
    parser.add_argument('--as-video', type=int, default=1)

    return parser.parse_args()


def main():
    args = parse_args()
    matches = os.listdir(args.movie_dir)
    for match in matches:
        ext = os.path.splitext(match)[-1]
        if ext == ".dat":
            print(match)
            loader=GodViewLoader({
                "globPattern":os.path.join(args.movie_dir, match),
                "outputPrefix":os.path.join(args.movie_dir, "movie"),
                "asVideo":args.as_video, # Trueだと動画(mp4)、Falseだと連番画像（png）として保存
                "fps":60
            })
            loader.run()

if __name__ == "__main__":
    main()
