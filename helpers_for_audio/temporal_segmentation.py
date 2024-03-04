import librosa
import soundfile
from pathlib import Path
from tqdm import tqdm
import os
import torch
import torchaudio
import torchaudio.transforms
import matplotlib.pylab as plt

# divides the audio to 5sec audios if duration is greater than 5 and returns the no of split segments


def temporal_segmentation_single_file(file: Path, output_dir: Path, segment_duration: int = 5) -> int:

    y, sr = librosa.load(file, sr=None)

    duration = librosa.get_duration(y=y, sr=sr)
    no_of_segments = 0
    if (duration > segment_duration):
        no_of_segments = int(duration//segment_duration)

        for i in range(no_of_segments):

            start_time = i*segment_duration
            end_time = (i+1)*segment_duration
            segment = y[int(start_time)*sr:int(end_time)*sr]
            segment_file = output_dir / \
                Path(f"{file.name.removesuffix('.wav')}_{i}.wav")
            try:
                soundfile.write(segment_file, segment, sr, format='wav')
            except:
                print(f"Error on file: {file}")
    return no_of_segments


def show_waveplot(file: Path):
    y, sr = librosa.load(file)
    librosa.display.waveshow(y, sr=sr, color='blue')


def temporal_segmentation_single_species(files: list[Path], output_dir: Path, segment_duration: int = 5) -> None:
    files_len = len(files)

    output_dir = output_dir / files[0].parent.name
    output_dir.mkdir(parents=True, exist_ok=True)
    file_count = 0
    for i in tqdm(range(files_len)):
        file_count += temporal_segmentation_single_file(
            files[i], output_dir=output_dir, segment_duration=segment_duration)

    print(f"Segmentation for: {output_dir.name} complete with {file_count} 5 sec segments")


def temporal_segmentation_all_data(parent_dir: Path, output_dir: Path, segment_duration: int = 5):
    birds_dir_path = [
        Path(parent_dir) / item for item in os.listdir(parent_dir)]
    wav_files_path = []

    for i in range(len(birds_dir_path)):
        wav_files_path.append(
            [birds_dir_path[i] / item for item in os.listdir(birds_dir_path[i])])

    len_birds = len(wav_files_path)
    for i in tqdm(range(len_birds)):
        temporal_segmentation_single_species(
            files=wav_files_path[i], output_dir=output_dir, segment_duration=segment_duration)


def plot_mfcc(mfcc):
    plt.imshow(mfcc, interpolation="nearest", origin="lower", aspect="auto")
    plt.colorbar()
