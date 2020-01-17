import Foundation
import TensorFlow
import TensorBoardX

let logdir = URL(fileURLWithPath: "/tmp/tensorboardx")
try? FileManager.default.removeItem(at: logdir)

// MARK: - Create SummaryWriter
let writer = SummaryWriter(logdir: logdir)

// MARK: - Add scalars to draw graph
for i in 0..<100 {
    writer.addScalar(tag: "scalar/scalar", scalar: Double(i), globalStep: i, date: Date(timeIntervalSince1970: Double(i)))
}

for i in 0..<100 {
    let sin = sinf(Float(i) / 30)
    let cos = cosf(Float(i) / 30)
    writer.addScalars(mainTag: "scalar/scalars", taggedScalars: ["sin": sin, "cos": cos], globalStep: i)
}

// MARK: - Add images
for i in 0..<3 {
    let image = Tensor<Double>(randomUniform: [128, 128, 3])
    writer.addImage(tag: "image", image: image, globalStep: i)
}

for i in 0..<3 {
    let images = Tensor<Float>(randomUniform: [5, 128, 128, 3])
    writer.addImages(tag: "images", images: images, globalStep: i)
}

// MARK: - Add texts
for i in 0..<3 {
    writer.addText(tag: "text", text: "step: \(i)", globalStep: i)
}
writer.addText(tag: "text_with_newlines", text: """
text
with
newlines
""")

// `Encodable`s can be written as JSON.
struct Obj: Encodable {
    var int: Int
    var text: String
}
let obj = Obj(int: 42, text: "hoge")
try writer.addJSONText(tag: "json", encodable: obj)

// MARK: - Add embeddings
let data = Tensor<Double>(randomNormal: [100, 10])
let labels = (0..<100).map { _ in String(Int.random(in: 0..<10)) }
//let labelImages = Tensor<Float>(randomNormal: [100, 3, 32, 32])

writer.addEmbedding(tag: "embed", matrix: data, labels: labels)

// MARK: - Add histograms to draw histograms/distributions
for i in 0..<3 {
    let hist = Tensor<Float>(randomNormal: [1024])
    writer.addHistogram(tag: "hist", values: hist, globalStep: i)
}

// If your model conforms to `HistogramWritable`, `addHistograms(tag: layer:)` is available.
let conv = Conv2D<Float>(filterShape: (3, 3, 32, 64))
writer.addHistograms(tag: "conv", layer: conv)

writer.flush()
