import Foundation
import TensorFlow
import TensorBoardX

try? FileManager.default.removeItem(atPath: "/tmp/tensorboardx")

let writer = SummaryWriter(logdir: "/tmp/tensorboardx")

for i in 0..<100 {
    writer.addScalar(tag: "scalar/scalar", scalar: Float(i), globalStep: i, date: Date(timeIntervalSince1970: Double(i)))
}

for i in 0..<100 {
    let sin = sinf(Float(i) / 30)
    let cos = cosf(Float(i) / 30)
    writer.addScalars(mainTag: "scalar/scalars", taggedScalars: ["sin": sin, "cos": cos], globalStep: i)
}

for i in 0..<3 {
    let image = Tensor<Float>(randomUniform: [128, 128, 3])
    writer.addImage(tag: "image", image: image, globalStep: i)
}

for i in 0..<3 {
    let images = Tensor<Float>(randomUniform: [5, 128, 128, 3])
    writer.addImages(tag: "images", images: images, globalStep: i)
}

for i in 0..<3 {
    let hist = Tensor<Float>(randomNormal: [1024])
    writer.addHistogram(tag: "hist", values: hist, globalStep: i)
}

for i in 0..<3 {
    writer.addText(tag: "text", text: "step: \(i)", globalStep: i)
}

let data = Tensor<Float>(randomNormal: [100, 10])
let labels = (0..<100).map { _ in String(Int.random(in: 0..<10)) }
//let labelImages = Tensor<Float>(randomNormal: [100, 3, 32, 32])

writer.addEmbedding(tag: "embed", matrix: data, labels: labels)

writer.close()
