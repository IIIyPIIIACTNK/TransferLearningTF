using Microsoft.ML;
using Microsoft.ML.Data;
using System.Data;

namespace TransferLearningTF
{
    internal class Program
    {


        static void Main(string[] args)
        {
            string _assetsPath = Path.Combine(Environment.CurrentDirectory, "assets");
            string _imagesFolder = Path.Combine(_assetsPath, "images");
            string _trainTagsTsv = Path.Combine(_imagesFolder, "tags.tsv");
            string _testTagsTsv = Path.Combine(_imagesFolder, "test-tags.tsv");
            string _predictSingleImage = Path.Combine(_imagesFolder, "toaster3.jpg");
            string _inceptionTensorFlowModel = Path.Combine(_assetsPath, "inception", "tensorflow_inception_graph.pb");


            MLContext mlContext = new MLContext();

            ITransformer model = GenerateModel(mlContext);

            ClassifySingleImage(mlContext, model);

            void DisplayResults(IEnumerable<ImagePrediction> imagePredictionData)
            {
                foreach (ImagePrediction prediction in imagePredictionData)
                {
                    Console.WriteLine($"Image: {Path.GetFileName(prediction.ImagePath)} predicted as: {prediction.PredictedLabelValue} with score: {prediction.Score?.Max()} ");
                }
            }

            void ClassifySingleImage(MLContext mlContext, ITransformer model)
            {
                var imageData = new ImageData()
                {
                    ImagePath = _predictSingleImage
                };

                var predictor = mlContext.Model.CreatePredictionEngine<ImageData, ImagePrediction>(model);
                var prediction = predictor.Predict(imageData);

                Console.WriteLine($"Image: {Path.GetFileName(imageData.ImagePath)} predicted as: {prediction.PredictedLabelValue} with score: {prediction.Score?.Max()} ");
            }

            ITransformer GenerateModel(MLContext mlContext)
            {
                // <SnippetImageTransforms>
                IEstimator<ITransformer> pipeline = mlContext.Transforms.LoadImages(outputColumnName: "input", imageFolder: _imagesFolder, inputColumnName: nameof(ImageData.ImagePath))
                                // The image transforms transform the images into the model's expected format.
                                .Append(mlContext.Transforms.ResizeImages(outputColumnName: "input", imageWidth: InceptionSettings.ImageWidth, imageHeight: InceptionSettings.ImageHeight, inputColumnName: "input"))
                                .Append(mlContext.Transforms.ExtractPixels(outputColumnName: "input", interleavePixelColors: InceptionSettings.ChannelsLast, offsetImage: InceptionSettings.Mean))
                                // </SnippetImageTransforms>
                                // The ScoreTensorFlowModel transform scores the TensorFlow model and allows communication
                                // <SnippetScoreTensorFlowModel>
                                .Append(mlContext.Model.LoadTensorFlowModel(_inceptionTensorFlowModel).
                                    ScoreTensorFlowModel(outputColumnNames: new[] { "softmax2_pre_activation" }, inputColumnNames: new[] { "input" }, addBatchDimensionInput: true))
                                // </SnippetScoreTensorFlowModel>
                                // <SnippetMapValueToKey>
                                .Append(mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "LabelKey", inputColumnName: "Label"))
                                // </SnippetMapValueToKey>
                                // <SnippetAddTrainer>
                                .Append(mlContext.MulticlassClassification.Trainers.LbfgsMaximumEntropy(labelColumnName: "LabelKey", featureColumnName: "softmax2_pre_activation"))
                                // </SnippetAddTrainer>
                                // <SnippetMapKeyToValue>
                                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabelValue", "PredictedLabel"))
                                .AppendCacheCheckpoint(mlContext);
                // </SnippetMapKeyToValue>


                //IEstimator<ITransformer> pipeline = mlContext.Transforms.LoadImages(outputColumnName: "input", imageFolder: _imagesFolder, inputColumnName: nameof(ImageData.ImagePath))
                //// The image transforms transform the images into the model's expected format.
                //    .Append(mlContext.Transforms.ResizeImages(outputColumnName: "input", imageWidth: InceptionSettings.ImageWidth, imageHeight: InceptionSettings.ImageHeight, inputColumnName: "input"))
                //    .Append(mlContext.Transforms.ExtractPixels(outputColumnName: "input", interleavePixelColors: InceptionSettings.ChannelsLast, offsetImage: InceptionSettings.Mean))
                //    .Append(mlContext.Model.LoadTensorFlowModel(_inceptionTensorFlowModel)
                //        .ScoreTensorFlowModel(outputColumnNames: new[] { "softmax2_pre_activation" }, inputColumnNames: new[] { "input" }, addBatchDimensionInput: true))
                //    .Append(mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "LabelKey", inputColumnName: "Label"))
                //    .Append(mlContext.MulticlassClassification.Trainers.LbfgsMaximumEntropy(labelColumnName: "LabelKey", featureColumnName: "softmax2_pre_activation"))
                //    .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabelValue", "PredictedLabel"))
                //        .AppendCacheCheckpoint(mlContext);

                IDataView trainingData = mlContext.Data.LoadFromTextFile<ImageData>(path: _trainTagsTsv, hasHeader: false);
                ITransformer model = pipeline.Fit(trainingData);

                IDataView testData = mlContext.Data.LoadFromTextFile<ImageData>(path: _testTagsTsv, hasHeader: false);
                IDataView predictions = model.Transform(testData);

                // Create an IEnumerable for the predictions for displaying results
                IEnumerable<ImagePrediction> imagePredictionData = mlContext.Data.CreateEnumerable<ImagePrediction>(predictions, true);
                DisplayResults(imagePredictionData);

                MulticlassClassificationMetrics metrics =
                    mlContext.MulticlassClassification.Evaluate(predictions,
                        labelColumnName: "LabelKey",
                        predictedLabelColumnName: "PredictedLabel");

                Console.WriteLine($"LogLoss is: {metrics.LogLoss}");
                Console.WriteLine($"PerClassLogLoss is: {String.Join(" , ", metrics.PerClassLogLoss.Select(c => c.ToString()))}");

                return model;
            }
        }
    }

    public class ImageData
    {
        [LoadColumn(0)]
        public string? ImagePath;

        [LoadColumn(1)]
        public string? Label;
    }

    public class ImagePrediction : ImageData
    {
        public float[]? Score;

        public string? PredictedLabelValue;
    }

    struct InceptionSettings
    {
        public const int ImageHeight = 224;
        public const int ImageWidth = 224;
        public const float Mean = 117;
        public const float Scale = 1;
        public const bool ChannelsLast = true;
    }
}
