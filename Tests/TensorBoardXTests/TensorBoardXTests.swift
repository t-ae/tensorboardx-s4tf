import XCTest
import TensorFlow
@testable import TensorBoardX

final class TensorBoardXTests: XCTestCase {
    func testExample() {
        try? FileManager.default.removeItem(atPath: "/tmp/tensorboardx")
        
        let writer = SummaryWriter(directory: "/tmp/tensorboardx")
        
        for i in 0..<100 {
            writer.addScalar(tag: "data/scalar1", scalar: Float(i), globalStep: i)
        }
        
        for i in 0..<3 {
            let image = Tensor<Float>(randomUniform: [128, 128, 3])
            writer.addImage(tag: "image", image: image, globalStep: i)
        }
        
        for i in 0..<3 {
            let images = Tensor<Float>(randomUniform: [5, 128, 128, 3])
            writer.addImages(tag: "images", images: images, globalStep: i)
        }
        
        writer.close()
    }

    static var allTests = [
        ("testExample", testExample),
    ]
}
