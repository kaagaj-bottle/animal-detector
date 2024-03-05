import os
from pathlib import Path
from pydub import AudioSegment
from tqdm import tqdm

INPUT_FORMAT = "mp3"
OUTPUT_FORMAT = "wav"


def convert_single_file_to_wav(file: Path, output_dir: Path):
    output_path = output_dir / Path(f"{file.name.removesuffix('.mp3')}.wav")

    try:
        sound = AudioSegment.from_file(file, format=INPUT_FORMAT)
    except:
        print(f"couldn't read the file: {file}")
    try:
        sound.export(output_path, format=OUTPUT_FORMAT)
    except:
        print(f"couldn't write the file: {file}")


def convert_single_species_to_wav(files: list[Path], output_dir: Path):
    files_len = len(files)
    if (files_len <= 0):
        return

    output_dir = output_dir / files[0].parent.name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"converting all files inside {files[0].parent.name}")
    for i in tqdm(range(files_len)):
        convert_single_file_to_wav(files[i], output_dir=output_dir)

# enter the dir path which has dirs in the following form
"""
    parent_dir
        -dir1
            -file1.mp3
            -file2.mp3
        -dir2
            -file3.mp3
            -file3.mp3
        -dir3
            -file1.mp3
            -file10.mp3
    just give the output_dir path 
"""
def convert_all_data_to_wav(parent_dir: Path, output_dir: Path):
    birds_dir_path = [
        Path(parent_dir) / item for item in os.listdir(parent_dir)]
    audio_files_path = []

    for i in range(len(birds_dir_path)):
        audio_files_path.append(
            [birds_dir_path[i] / item for item in os.listdir(birds_dir_path[i])])

    len_audios = len(audio_files_path)

    for i in range(len_audios):
        convert_single_species_to_wav(audio_files_path[i], output_dir)
