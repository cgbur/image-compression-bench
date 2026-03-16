# Image Compression Benchmark

[Blog post](https://cgbur.com/posts/image-compression/)

Benchmarks AVIF, JPEG XL, WebP, and JPEG across quality levels and resolutions. Measures perceptual quality using [SSIMULACRA 2](https://github.com/cloudinary/ssimulacra2) to find the Pareto frontier of file size vs visual quality.

## Usage

```sh
python bench.py sources/photo1.avif sources/photo2.jpg
python bench.py sources/*.avif -o results/
```

Put source images in `sources/` (gitignored). Results go to `results/` (also gitignored). Plots are saved as PNGs.

## Requirements

- Python 3.10+
- ImageMagick 7+ with AVIF and JXL support
- [SSIMULACRA 2](https://github.com/cloudinary/ssimulacra2) binary at `./ssimulacra2` (see below)
- matplotlib

## Building SSIMULACRA 2

Clone and build from source (requires cmake, ninja, libhwy, lcms2, libjpeg, libpng):

```sh
git clone --depth 1 https://github.com/cloudinary/ssimulacra2.git /tmp/ssimulacra2
cd /tmp/ssimulacra2
mkdir build && cd build
cmake ../src -G Ninja -DCMAKE_BUILD_TYPE=Release
ninja ssimulacra2
strip ssimulacra2
cp ssimulacra2 /path/to/image-compression-bench/
```

With Nix:

```sh
nix-shell -p cmake ninja pkg-config libhwy lcms2 libjpeg libpng --run "
  cd /tmp/ssimulacra2 && mkdir -p build && cd build &&
  cmake ../src -G Ninja -DCMAKE_BUILD_TYPE=Release &&
  ninja ssimulacra2 && strip ssimulacra2
"
cp /tmp/ssimulacra2/build/ssimulacra2 .
```
