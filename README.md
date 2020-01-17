# TensorBoardX

[tensorboardX](https://github.com/lanpa/tensorboardX) wrapper for Swift for TensorFlow.

## Install

### Requirements

- [Swift for TensorFlow toolchain](https://github.com/tensorflow/swift/blob/master/Installation.md)
- Python libraries

```bash
$ pip install tensorboard tensorboardX numpy Pillow
```

To verify installation, clone this repository and run the commands below.

```bash
$ swift run TestPlot
$ tensorboard --logdir /tmp/tensorboardx
```

### SwiftPM

Add this to package's dependencies.

```
.package(url: "https://github.com/t-ae/tensorboardx-s4tf.git", from: "0.0.8"),
```

And add `"TensorBoardX"` for target's dependencies.

## Example
[Sources/TestPlot/main.swift](https://github.com/t-ae/tensorboardx-s4tf/blob/master/Sources/TestPlot/main.swift)