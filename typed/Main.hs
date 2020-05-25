module Main where

import Torch.HList
import Torch.Typed.Parameter ( MakeIndependent(..), ToDependent(..), replaceParameters, flattenParameters)
import Torch.Typed.Tensor
import Torch.Typed.Serialize (load)
import Torch.Typed.Optim (mkAdam)
import Torch.Typed.NN (LinearSpec(..), Linear(..), linear)
import Torch.Typed.Functional (upsample_bilinear2d, expand)
import qualified Torch.NN as A
import qualified Torch.Device as D
import qualified Torch.DType as D
import GHC.TypeLits
import           GHC.TypeLits.Extra
import qualified Torch.Typed.Vision as I

import Typed.Vgg
import Common


--   TODO: I think I need to make a hlist of all the weights in vgg so I can deserialize it :(
type BatchSize = 12
forward final vggParams = linear final . vgg16ForwardNoFinal vggParams

train' ::  
  (Tensor '(D.CUDA, 0) 'D.Float '[BatchSize, I.DataDim] ->
   Tensor '(D.CUDA, 0) 'D.Float  '[BatchSize, 3, 224, 224]) ->
  Vgg16 'D.Float '(D.CUDA, 0) -> IO ()
  --  (Tensor device dtype shape ->
  --  Tensor device dtype shape') ->
  -- Vgg16 dtype device -> IO ()
train' transforms vggParams = do
  (model :: Linear 4096 10 'D.Float '(D.CUDA, 0)) <- A.sample (LinearSpec @4096 @10) 
  let initOptim = mkAdam 0 0.9 0.99 (flattenParameters model)
  print "hey"
  train @BatchSize @'(D.CUDA, 0) model initOptim (\model _ input -> pure $ forward model vggParams (transforms input)) 0.1 "vgg16-mnist.pt"

imageTransform =  upsample_bilinear2d @224 @224 False . (expand @'[BatchSize, 3, 28, 28] False) . reshape @'[BatchSize, 1, 28, 28]
main = do
  -- TODO: get the types to align when loading in weights
  -- tensors <-  load @(Tensor '(D.CPU, 0) 'D.Float '[0] )"vgg16.pt" 
  -- let vgg = hmap' ToDependent tensors
  vggRandom <- A.sample (VggSpec @'D.Float @'(D.CUDA, 0))
  -- vggParams <- hmapM' MakeIndependent vgg
  -- let x = replaceParameters vggRandom vggParams

  train' imageTransform vggRandom
