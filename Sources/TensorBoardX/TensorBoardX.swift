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
    
    deinit {
        close()
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
        precondition(image.rank == 3, "Invalid `images` rank.")
        switch dataformats {
        case .channelsFirst:
            precondition([1, 3, 4].contains(image.shape[0]), "Invalid `image` shape.")
            precondition(image.shape[1] > 0 && image.shape[2] > 0, "Invalid `image` shape.")
        case .channelsLast:
            precondition([1, 3, 4].contains(image.shape[2]), "Invalid `image` shape.")
            precondition(image.shape[0] > 0 && image.shape[1] > 0, "Invalid `image` shape.")
        }
        writer.add_image(tag: tag,
                         img_tensor: nparray(image),
                         global_step: globalStep,
                         walltime: date?.walltime,
                         dataformats: dataformats.stringValue)
    }
    
    /// Add images to summary.
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
        precondition(images.rank == 4, "Invalid `images` rank.")
        precondition(images.shape[0] > 0, "`images` contains no images.")
        switch dataformats {
        case .channelsFirst:
            precondition([1, 3, 4].contains(images.shape[1]), "Invalid `images` shape.")
            precondition(images.shape[2] > 0 && images.shape[3] > 0, "Invalid `image` shape.")
        case .channelsLast:
            precondition([1, 3, 4].contains(images.shape[3]), "Invalid `images` shape.")
            precondition(images.shape[1] > 0 && images.shape[2] > 0, "Invalid `image` shape.")
        }
        let dataformats = "N" + dataformats.stringValue
        writer.add_images(tag: tag,
                          img_tensor: nparray(images),
                          global_step: globalStep,
                          walltime: date?.walltime,
                          dataformats: dataformats)
    }
    
    /// Add images as grid with specified `colsize` to summary.
    /// - Parameters:
    ///   - images: Tensor contains images. Number of channels must be 1, 3, or 4. Pixel values must be in [0, 1] range.
    ///   - colsize: The number of images each row contains.
    ///   - dataformats: Specify where channels dimension is in `images` tensor dimensions.
    public func addImages<T: FloatingPoint&PythonConvertible>(
        tag: String,
        images: Tensor<T>,
        colSize: Int,
        dataformats: ImageDataFormat = .channelsLast,
        globalStep: Int? = nil,
        date: Date? = nil
    ) {
        precondition(images.rank == 4, "Invalid `images` rank.")
        let (s1, s2, s3) = (images.shape[1], images.shape[2], images.shape[3])
        let paddings = colSize - images.shape[0] % colSize
        let padImages = Tensor<T>(zeros: [paddings, s1, s2, s3])
        
        var grid = Tensor(concatenating: [images, padImages], alongAxis: 0)
        let rowSize = grid.shape[0] / colSize
        grid = grid.reshaped(to: [rowSize, colSize, s1, s2, s3])
        switch dataformats {
        case .channelsFirst:
            grid = grid.transposed(permutation: 0, 3, 1, 4, 2)
            grid = grid.reshaped(to: [rowSize*s2, colSize*s3, s1])
        case .channelsLast:
            grid = grid.transposed(permutation: 0, 2, 1, 3, 4)
            grid = grid.reshaped(to: [rowSize*s1, colSize*s2, s3])
        }
        addImage(tag: tag, image: grid, dataformats: dataformats)
    }
    
    /// Add text to summary.
    /// - Parameters:
    ///   - convertToMarkDownFormat: If true, newlines in text will be converted to "  \n".
    public func addText(tag: String,
                        text: String,
                        globalStep: Int? = nil,
                        date: Date? = nil,
                        convertToMarkdownFormat: Bool = true) {
        var text = text
        if convertToMarkdownFormat {
            text = text.replacingOccurrences(of: "\n", with: "  \n")
        }
        writer.add_text(tag: tag,
                        text_string: text,
                        global_step: globalStep,
                        walltime: date?.walltime)
    }
    
    /// Add json text to summary.
    /// - Parameters:
    ///   - encoder: `JSONEncoder` to use. Default is prettyPrinted enabled.
    public func addJSONText<T: Encodable>(
        tag: String,
        encodable: T,
        globalStep: Int? = nil,
        date: Date? = nil,
        encoder: JSONEncoder = {
            let encoder = JSONEncoder()
            encoder.outputFormatting = .prettyPrinted
            return encoder
        }()
    ) throws {
        var text = try String(data: encoder.encode(encodable), encoding: .utf8)!
        text = text.replacingOccurrences(of: " ", with: "&nbsp;")
        addText(tag: tag, text: text, globalStep: globalStep, date: date)
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
    public func addEmbedding<T: FloatingPoint&PythonConvertible>(
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
