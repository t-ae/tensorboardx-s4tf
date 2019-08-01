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
    
    /// Flush.
    public func flush() {
        writer.flush()
    }
    
    /// Close.
    public func close() {
        writer.close()
    }
}

extension SummaryWriter {
    /// Add scalar to summary.
    public func addScalar(tag: String, scalar: Float, globalStep: Int? = nil) {
        writer.add_scalar(tag, scalar, globalStep)
    }
    
    /// Add scalars to summary.
    public func addScalars(mainTag: String,
                           taggedScalars: [String: Float],
                           globalStep: Int? = nil) {
        writer.add_scalars(mainTag, taggedScalars, globalStep)
    }
    
    /// Add image to summary.
    /// - Parameters:
    ///   - image: Image tensor. Number of channels must be 1, 3, or 4. Pixel values must be in [0, 1] range.
    ///   - dataformats: Specify where channels dimension is in `images` tensor dimensions.
    public func addImage(tag: String,
                         image: Tensor<Float>,
                         globalStep: Int? = nil,
                         dataformats: ImageDataFormat = .channelsLast) {
        precondition(image.shape.count == 3, "Invalid `images` shape.")
        switch dataformats {
        case .channelsFirst:
            precondition([0, 3, 4].contains(image.shape[0]), "Invalid `image` shape.")
            precondition(image.shape[1] > 0 && image.shape[2] > 0, "Invalid `image` shape.")
        case .channelsLast:
            precondition([0, 3, 4].contains(image.shape[2]), "Invalid `image` shape.")
            precondition(image.shape[0] > 0 && image.shape[1] > 0, "Invalid `image` shape.")
        }
        writer.add_image(tag, nparray(image), globalStep, dataformats: dataformats.stringValue)
    }
    
    /// Add images to summary
    /// - Parameters:
    ///   - images: Tensor contains images. Number of channels must be 1, 3, or 4. Pixel values must be in [0, 1] range.
    ///   - dataformats: Specify where channels dimension is in `images` tensor dimensions.
    public func addImages(tag: String,
                          images: Tensor<Float>,
                          globalStep: Int? = nil,
                          dataformats: ImageDataFormat = .channelsLast) {
        precondition(images.shape.count == 4, "Invalid `images` shape.")
        precondition(images.shape[0] > 0, "`images` contains no images.")
        switch dataformats {
        case .channelsFirst:
            precondition([0, 3, 4].contains(images.shape[1]), "Invalid `images` shape.")
            precondition(images.shape[2] > 0 && images.shape[3] > 0, "Invalid `image` shape.")
        case .channelsLast:
            precondition([0, 3, 4].contains(images.shape[3]), "Invalid `images` shape.")
            precondition(images.shape[1] > 0 && images.shape[2] > 0, "Invalid `image` shape.")
        }
        let dataformats = "N" + dataformats.stringValue
        writer.add_images(tag, nparray(images), globalStep, dataformats: dataformats)
    }
    
    /// Add text to summary.
    public func addText(tag: String, text: String, globalStep: Int? = nil) {
        writer.add_text(tag, text, globalStep)
    }
    
    /// Add histogram to summary.
    public func addHistogram(tag: String, values: Tensor<Float>, globalStep: Int? = nil) {
        writer.add_histogram(tag, nparray(values), globalStep)
    }
    
    /// Add embedding.
    /// - Parameters:
    ///   - matrix: N x D matrix, N features of D dimension.
    ///   - labels: Labels for each sample.
    public func addEmbedding(tag: String = "default",
                             matrix: Tensor<Float>,
                             labels: [String],
                             globalStep: Int? = nil) {
        writer.add_embedding(mat: nparray(matrix),
                             metadata: labels,
                             global_step: globalStep,
                             tag: tag)
    }
    
    /// Add embedding.
    /// - Parameters:
    ///   - matrix: N x D matrix, N features of D dimension.
    ///   - labels: Labels for each sample.
    // Currently unavailable since label_img is PyTorch tensor only.
    private func addEmbedding(tag: String = "default",
                             matrix: Tensor<Float>,
                             labels: [String],
                             labelImages: Tensor<Float>,
                             globalStep: Int? = nil) {
        writer.add_embedding(mat: nparray(matrix),
                             metadata: labels,
                             label_img: nparray(labelImages),
                             global_step: globalStep,
                             tag: tag)
    }
}
