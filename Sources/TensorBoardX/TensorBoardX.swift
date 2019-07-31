import TensorFlow
import Python

let np = Python.import("numpy")
let tbx = Python.import("tensorboardX")

func nparray(_ tensor: Tensor<Float>) -> PythonObject {
    return np.array(tensor.scalars).reshape(tensor.shape)
}

public class SummaryWriter {
    var writer: PythonObject
    
    public init(directory: String) {
        self.writer = tbx.SummaryWriter(directory)
    }
}

extension SummaryWriter {
    public func addScalar(tag: String, scalar: Float, globalStep: Int) {
        writer.add_scalar(tag, scalar, globalStep)
    }
    
    public func addImage(tag: String,
                         image: Tensor<Float>,
                         globalStep: Int,
                         dataformats: String = "HWC") {
        writer.add_image(tag, nparray(image), globalStep, dataformats: dataformats)
    }
    
    public func addImages(tag: String,
                          images: Tensor<Float>,
                          globalStep: Int,
                          dataformats: String = "NHWC") {
        writer.add_images(tag, nparray(images), globalStep, dataformats: dataformats)
    }
    
    public func close() {
        writer.close()
    }
}
