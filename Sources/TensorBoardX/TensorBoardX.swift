import TensorFlow
import Python

let np = Python.import("numpy")
let tbx = Python.import("tensorboardX")

func nparray(_ tensor: Tensor<Float>) -> PythonObject {
    return np.array(tensor.scalars).reshape(tensor.shape)
}

public enum ImageDataFormat {
    case channelsFirst, channelsLast
    
    var stringValue: String {
        switch self {
        case .channelsFirst:
            return "CHW"
        case .channelsLast:
            return "HWC"
        }
    }
}

public class SummaryWriter {
    var writer: PythonObject
    
    public init(logdir: String, flushSecs: Int = 120) {
        self.writer = tbx.SummaryWriter(logdir, flush_secs: flushSecs)
    }
}

extension SummaryWriter {
    /// Add scalar to summary.
    public func addScalar(tag: String, scalar: Float, globalStep: Int) {
        writer.add_scalar(tag, scalar, globalStep)
    }
    
    /// Add scalars to summary.
    public func addScalars(tag: String, scalars: [String: Float], globalStep: Int) {
        writer.add_scalars(tag, scalars, globalStep)
    }
    
    /// Add image to summary.
    public func addImage(tag: String,
                         image: Tensor<Float>,
                         globalStep: Int,
                         dataformats: ImageDataFormat = .channelsLast) {
        writer.add_image(tag, nparray(image), globalStep, dataformats: dataformats.stringValue)
    }
    
    /// Add images to summary
    public func addImages(tag: String,
                          images: Tensor<Float>,
                          globalStep: Int,
                          dataformats: ImageDataFormat = .channelsLast) {
        let dataformats = "N" + dataformats.stringValue
        writer.add_images(tag, nparray(images), globalStep, dataformats: dataformats)
    }
    
    /// Add text to summary.
    public func addText(tag: String, text: String, globalSteps: Int) {
        writer.add_text(tag, text, globalSteps)
    }
    
    /// Add histogram to summary.
    public func addHistogram(tag: String, values: Tensor<Float>, globalSteps: Int) {
        writer.add_histogram(tag, nparray(values), globalSteps)
    }
    
    /// Flush.
    public func flush() {
        writer.flush()
    }
    
    /// Close.
    public func close() {
        writer.close()
    }
}
