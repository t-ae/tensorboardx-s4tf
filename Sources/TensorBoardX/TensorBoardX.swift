import Foundation
import TensorFlow
import Python

let np = Python.import("numpy")
let tbx = Python.import("tensorboardX")

func nparray<T: FloatingPoint&PythonConvertible>(_ tensor: Tensor<T>) -> PythonObject {
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
    
    public init(logdir: URL, flushSecs: Int = 120) {
        self.writer = tbx.SummaryWriter(logdir.path, flush_secs: flushSecs)
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

extension Date {
    var walltime: Float {
        return Float(timeIntervalSince1970)
    }
}

extension SummaryWriter {
    /// Add scalar to summary.
    public func addScalar<T: FloatingPoint&PythonConvertible>(
        tag: String,
        scalar: T,
        globalStep: Int? = nil,
        date: Date? = nil
    ) {
        writer.add_scalar(tag: tag,
                          scalar_value: scalar,
                          global_step: globalStep,
                          walltime: date?.walltime)
    }
    
    /// Add scalars to summary.
    public func addScalars<T: FloatingPoint&PythonConvertible>(
        mainTag: String,
        taggedScalars: [String: T],
        globalStep: Int? = nil,
        date: Date? = nil
    ) {
        writer.add_scalars(main_tag: mainTag,
                           tag_scalar_dict: taggedScalars,
                           global_step: globalStep,
                           walltime: date?.walltime)
    }
    
    /// Add image to summary.
    /// - Parameters:
    ///   - image: Image tensor. Number of channels must be 1, 3, or 4. Pixel values must be in [0, 1] range.
    ///   - dataformats: Specify where channels dimension is in `images` tensor dimensions.
    public func addImage<T: FloatingPoint&PythonConvertible>(
        tag: String,
        image: Tensor<T>,
        dataformats: ImageDataFormat = .channelsLast,
        globalStep: Int? = nil,
        date: Date? = nil
    ) {
        precondition(image.shape.count == 3, "Invalid `images` shape.")
        switch dataformats {
        case .channelsFirst:
            precondition([0, 3, 4].contains(image.shape[0]), "Invalid `image` shape.")
            precondition(image.shape[1] > 0 && image.shape[2] > 0, "Invalid `image` shape.")
        case .channelsLast:
            precondition([0, 3, 4].contains(image.shape[2]), "Invalid `image` shape.")
            precondition(image.shape[0] > 0 && image.shape[1] > 0, "Invalid `image` shape.")
        }
        writer.add_image(tag: tag,
                         img_tensor: nparray(image),
                         global_step: globalStep,
                         walltime: date?.walltime,
                         dataformats: dataformats.stringValue)
    }
    
    /// Add images to summary
    /// - Parameters:
    ///   - images: Tensor contains images. Number of channels must be 1, 3, or 4. Pixel values must be in [0, 1] range.
    ///   - dataformats: Specify where channels dimension is in `images` tensor dimensions.
    public func addImages<T: FloatingPoint&PythonConvertible>(
        tag: String,
        images: Tensor<T>,
        dataformats: ImageDataFormat = .channelsLast,
        globalStep: Int? = nil,
        date: Date? = nil
    ) {
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
        writer.add_images(tag: tag,
                          img_tensor: nparray(images),
                          global_step: globalStep,
                          walltime: date?.walltime,
                          dataformats: dataformats)
    }
    
    /// Add text to summary.
    public func addText(tag: String,
                        text: String,
                        globalStep: Int? = nil,
                        date: Date? = nil) {
        writer.add_text(tag: tag,
                        text_string: text,
                        global_step: globalStep,
                        walltime: date?.walltime)
    }
    
    /// Add histogram to summary.
    public func addHistogram<T: FloatingPoint&PythonConvertible>(
        tag: String,
        values: Tensor<T>,
        globalStep: Int? = nil,
        date: Date? = nil
    ) {
        writer.add_histogram(tag: tag,
                             values: nparray(values),
                             global_step: globalStep,
                             walltime: date?.walltime)
    }
    
    /// Add embedding.
    /// - Parameters:
    ///   - matrix: N x D matrix, N features of D dimension.
    ///   - labels: Labels for each sample.
    public func addEmbedding<T: FloatingPoint&PythonConvertible>(
        tag: String = "default",
        matrix: Tensor<T>,
        labels: [String],
        globalStep: Int? = nil
    ) {
        writer.add_embedding(mat: nparray(matrix),
                             metadata: labels,
                             global_step: globalStep,
                             tag: tag)
    }
    
    /// Add embedding.
    /// - Parameters:
    ///   - matrix: N x D matrix, N features of D dimension.
    ///   - labels: Labels for each sample.
    private func addEmbedding<T: FloatingPoint&PythonConvertible>(
        tag: String = "default",
        matrix: Tensor<T>,
        labels: [String],
        labelImages: Tensor<T>,
        globalStep: Int? = nil
    ) {
        writer.add_embedding(mat: nparray(matrix),
                             metadata: labels,
                             label_img: nparray(labelImages),
                             global_step: globalStep,
                             tag: tag)
    }
}
