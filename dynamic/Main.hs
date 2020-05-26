module Main where

import Torch
import Control.Monad (when, forM_)
import System.Random (mkStdGen, randoms)

import Torch.Functional as F
import qualified Torch.Vision as V
import qualified Torch.Typed.Vision as V hiding (getImages')
import           Torch.Serialize as S
import GHC.Generics

import Vgg

randomIndexes :: Int -> [Int]
randomIndexes size = (`mod` size) <$> randoms seed where seed = mkStdGen 123


resizeImage batchSize img = expand upsampled  True [batchSize, 3,224,224] 
  where upsampled = upsampleBilinear2d (224,224) False  $ reshape [batchSize,1,28,28] img

finalLayer :: LinearSpec
finalLayer = LinearSpec 4096 10

forward :: Vgg -> Linear -> Tensor ->  Tensor
forward vggParams state img = logSoftmax (Dim 1) $ linear state $ vgg16NoFinal vggParams img 

withCuda = withDevice gpu defaultOpts
gpu = Device CUDA 0 
cpu = Device CPU 0 

train :: Vgg -> V.MnistData -> IO Linear
train vggParams trainData = do
    spec <- sample finalLayer
    let params =  flattenParameters spec
    let toCuda = (toDevice gpu . toDependent) <$> params
    newParams <- sequenceA $ fmap makeIndependent toCuda
    let optimizer = mkAdam 0 0.9 0.99 newParams
    let init =  replaceParameters spec newParams
    let nImages = V.length trainData
        idxList = randomIndexes nImages
    trained <- foldLoop init numIters $
        \state iter -> do
            let idx = take batchSize (drop (iter * batchSize) idxList)
            input <- V.getImages' batchSize dataDim trainData idx
            let rszd = toDevice gpu $ normalize batchSize $ resizeImage batchSize input 
                label = toDevice gpu $ V.getLabels' batchSize trainData idx
                loss = nllLoss' label $  forward vggParams state rszd
            when (iter `mod` 50 == 0) $ do
                putStrLn $ "Iteration: " ++ show iter ++ " | Loss: " ++ show loss
            (newParam, _) <- runStep state optimizer loss 1e-4
            pure $ replaceParameters state newParam
    pure trained
    where
      dataDim = 784
      numIters = 3000
      batchSize = 16


main :: IO ()
main = do
    (trainData, testData) <- V.initMnist "data"
    vggRand <- sample vgg16Spec 
    vggCpu <- S.loadParams vggRand "build/vgg16.pt" 
    let flat = flattenParameters vggCpu 
    let toCuda = (toDevice gpu . toDependent) <$> flat
    newParams <- sequenceA $ fmap makeIndependent toCuda
    let vggParams = replaceParameters vggCpu newParams 
    model <- train vggParams trainData
    
    mapM (\idx -> do
        testImg <- toDevice gpu <$> V.getImages' 1 784 testData [idx]
        V.dispImage (toDevice cpu $ testImg)
        let rszd = normalize 1 $ resizeImage 1 testImg
        putStrLn $ "Model        : " ++ (show . toDevice cpu. (argmax (Dim 1) RemoveDim) .
                                         Torch.exp $
                                         forward vggParams model rszd
                                        )

        putStrLn $ "Ground Truth : " ++ (show $ V.getLabels' 1 testData [idx])
        ) [0..10]

    putStrLn "Done"
