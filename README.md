# Scalable Video Coding Based on Content and Gaze

![License](https://img.shields.io/github/license/fonzcastellanos/scalable-video-codec)

## Overview

This project is an exploration of scalable video coding based on content (i.e. physical objects) and point of gaze (i.e. where a viewer is looking). Scalability, in this context, is the ability to reconstruct meaningful video information from partial decompressed streams, thereby helping video systems meet their client device processing power and network bandwidth requirements.

The encoder first performs block-based motion estimation and then segments the video into regions, which are sets of spatially-connected blocks with similar motion. Each region represents either the background or a distinct moving (i.e. foreground) object. Afterwards, the encoder applies the Discrete Cosine Transform (DCT) and the resulting coefficients and identifier of each region are written to a file.

<video src="https://user-images.githubusercontent.com/4334520/219800414-400cd0c6-75cd-4a47-95ed-9866ebfcdd7e.mov"></video>

In a typical streaming architecture, each region would be converted into quantized DCT coefficients based on bandwidth and gaze requirements, and the compressed data would be buffered for streaming. In such architectures, the continuous bandwidth sensing would control the quantization of the DCT coefficients of each region, with background transform blocks generally being more quantized than foreground transform blocks. The gaze requirements determine which transform blocks need to be more clear, with transform blocks in the gaze areas being less quantized than those outside of it. In these cases, both properties of bandwidth and gaze control are communicated by the client to the streaming encoder.

For the sake of simplicity, I emulate the streaming feature. The encoder computes and stores all the transform coefficients, but the decoder will decide on the degree of quantization to apply based on whether a region part of the foreground or is the background and where the user is gazing. This approach allows me to focus on block-based motion estimation and segmentation without tackling the the complexities of streaming. I also emulate the gaze point, which is determined by the position of the mouse cursor.

<video src="https://user-images.githubusercontent.com/4334520/219826986-723024b9-adc3-48d6-8ac7-5541b2009453.mov"></video>

## Video Model 
- Illumination model: constant intensity assumption
    - Valid for spatially and temporally invariant ambient illumination sources and diffuse reflecting surfaces
    - The surface reflectance of an object does not change as the object moves
    - Describes neither moving shadows nor reflections due to glossy surfaces
- Scene model: 2D
    - All objects are flat and lay on the same image plane
    - Objects are limited to moving in a 2D plane
- Motion model: 2D
    - Motion is translational
    - Applies to camera and objects

## Encoder Steps
1. Convert frame from the RGB color space to the YUV color space
2. Extract Y (i.e. luminance) channel
    - All motion estimation is intensity-based and relies solely on the Y channel
3. Estimate block-wise motion
    - Achieved by a variation of the hierarchical block matching algorithm (HBMA)
        - For more details, open the file [`motion.hpp`](motion.hpp) and read the comment block at the top of the file and the comment block for the function `EstimateMotionHierarchical`
4. Estimate global motion using random sample consensus (RANSAC)
    - Global motion is assumed to be the camera motion
    - Inlier group is assumed to be the background motion vectors
    - Outlier group is assumed to be the foreground motion vectors
    - For more details, open the file [`motion.hpp`](motion.hpp) and read the comment block at the top of the file and the comment block for the function `EstimateGlobalMotionRansac`
5. Create foreground mask from outlier group
6. Improve spatial connectivity and remove noise in the foreground mask
    - Achieved by applying the closing and opening morphological operators in the stated order
        - Structural element is rectangular
7. Segment foreground layer into regions
    1. Apply k-means on foreground layer
        - Each feature vector consists of motion vector position and components
        - A cluster is not guaranteed to be spatially connected. Step 7.2 addresses this
    2. Find the connected components of each cluster
        - Each connected component becomes a region
8. Compute the Discrete Cosine Transform (DCT)
    - The frame in the RGB color space is divided into transform blocks, which undergo the DCT
    - Each channel is processed independently
9. Write encoded frames to a file
    - Each transform block is assigned a block type
    - The background and each foreground region is mapped to a unique block type
    - For every transform block, the block type and the DCT coefficients for each channel are written

## Decoder Steps
1. Emulate bandwidth scalability and gaze control
    - If a block is in the gaze region, then no quantization occurs and substeps below are skipped
    1. The DCT coefficients of a block are quantized by dividing by either `foreground-quant-step` or `background-quant-step` (see [`decoder` usage section](#decoder)), depending on whether the block belongs to the foreground or background
    2. The resulting quantized values are rounded off, and the reverse process is applied to obtain the dequantized DCT coefficients
2. Compute the inverse DCT of dequantized blocks

## Results

The videos I used for testing were from the 2014 IEEE Change Detection Workshop (CDW-2014) dataset, which can be found here: http://changedetection.net/. 

The names of the result videos follow the format `<executable-name>_<video-category>_<video-name>.mov`.

For example, the name of the videos used in the [overview](#overview) are 
- `encoder-visualizer_dynamic-background_fall.mov`
- `decoder_dynamic-background_fall.mov`.

<video src="https://user-images.githubusercontent.com/4334520/219816152-e6fad12e-f218-4f15-a645-ce3a7e3c101f.mov"></video>

<video src="https://user-images.githubusercontent.com/4334520/219827606-9e425072-3e89-4a0d-b223-54c9301106c3.mov"></video>


<video src="https://user-images.githubusercontent.com/4334520/219825739-d47cdb67-67d7-4bb4-b083-ba9e05750f17.mov"></video>

<video src="https://user-images.githubusercontent.com/4334520/219827931-12d95fc5-3b66-4e92-8e19-7cc6fe0c65e2.mov"></video>

## Build Requirements
- C++ compiler
    - Must support at least C++17
    - I used Clang (version 10.0.01)
- CMake >= 3.2
    - I used 3.25.1
- OpenCV == 3.4.*
    - I used 3.4.16

I have not built the project on other platforms besides the one used for my development environment:

- Hardware: MacBook Pro (Retina, 13-inch, Early 2015)
- Operating System: macOS Mojave (version 10.14.16)

However, there is a good chance that the project can be built on your platform without much hassle because I use CMake to generate the build system. CMake is best known for taking a generic project description (see file [`CMakeLists.txt`](CMakeLists.txt)) and generating a platform-specific build system. 

To install OpenCV, follow the [OpenCV installation instructions](https://docs.opencv.org/3.4.16/df/d65/tutorial_table_of_content_introduction.html) for your platform. Once installed, CMake can use the command `find_package` to find the OpenCV package and load its package-specific details. 

## Build Steps
The following build steps assume a POSIX-compliant shell. 

The build system must be generated before the project can be built. From the project directory, generate the build system.
```sh
mkdir build
cd build
cmake ..
```
The `-G` option is omitted in `cmake ..`, so CMake will choose a default build system generator type based on your platform. To learn more about generators, see the [CMake docs](https://cmake.org/cmake/help/latest/manual/cmake-generators.7.html).

Go back to the project directory
```sh
cd ..
```
and build *all* the targets of the project. 
```sh
cmake --build build --config Release
```

The built targets are placed in the directory `build`.

The executable targets are
- encoder-visualizer
- encoder
- decoder

To build only a specific target, replace `mytarget` with the name of the target and execute:
```sh
cmake --build build --config Release --target mytarget
```

## Usage
Each option provided at the command-line must have its name prefixed with "--" and have an associated argument following its name. Options must also be before positional parameters. 

For example, the option `kmeans-cluster-count`, its associated argument of 12, and the video file path `foreman.mp4`, a positional parameter, would be passed to `encoder` like so. 
```sh
encoder --kmeans-cluster-count 12 foreman.mp4
```

### encoder
`encoder` writes the encoded video to the standard output stream `stdout`. 

To run `encoder` with the default configuration and write the encoded video to a file, execute the following command, which redirects output from `stdout` to `encoded_video_file_path`.
```sh
./build/encoder video_file_path > encoded_video_file_path
```

If you do not want to create a encoded video file but you still want to run the `decoder` on the `encoder` output, run `encoder` and `decoder` concurrently and connect the `stdout` of `encoder` to the `stdin` of `decoder`. Achieve this by executing the following command.
```sh
./build/encoder video_file_path | ./build/decoder
```

### encoder-visualizer
To run `encoder-visualizer` with the default configuration, execute the following command.
```sh
./build/encoder-visualizer video_file_path
```
### encoder & encoder-visualizer

`encoder` and `encoder-visualizer` have the same options.

To see the name and type of each option, search for `#options` in [`encoder_config.cpp`](encoder_config.cpp). You'll see an array called `opts` in the function `ParseConfig`. Each element of the array corresponds to an option and contains the name and type of the option.

To see the default values of the options, search for `#default-cfg` in [`encoder_config.cpp`](encoder_config.cpp). You'll see some functions containing the default values.

### decoder
The `decoder` reads encoded video from the standard input stream `stdin`. 

To run `decoder` with the default configuration and read encoded video from a file, execute the following command, which redirects input from `stdin` to `encoded_video_file_path`.
```sh
./build/decoder < encoded_video_file_path
```

If you want to run `decoder` on the output of `encoder` without creating an encoded video file, run `encoder` and `decoder` concurrently and connect the `stdout` of `encoder` to the `stdin` of `decoder`. Achieve this by executing the following command.
```sh
./build/encoder video_file_path | ./build/decoder
```

To see the name and type of each option, search for `#options` in [`decoder_config.cpp`](decoder_config.cpp). You'll see an array called `opts` in the function `ParseConfig`. Each element of the array corresponds to an option and contains the name and type of the option.

To see the default values of the options, search for `#default-cfg` in [`decoder_config.cpp`](decoder_config.cpp). You'll see some functions containing the default values.

## Future Direction
- Address oversegmentation by merging regions
- Implement entropy coding
- Derive prediction error images and compress them using the JPEG pipeline
- Compress motion vectors using entropy coding
- Adaptive tuning of certain parameters
- Do I/O on a separate thread
- Use SIMD instructions to vectorize motion estimation algorithms
- Implement streaming
- Eliminate dependence on OpenCV

## References
- Multimedia Systems: Algorithms, Standards, and Industry Practices
    - By Parag Havaldar and Gerard Medioni
- Video Processing and Communications
    - By Yao Wang, Jôrn Ostermann, and Ya-Qin Zhang
