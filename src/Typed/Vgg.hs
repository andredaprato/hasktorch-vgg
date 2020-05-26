{-# LANGUAGE NoStarIsType #-}
module Typed.Vgg where

import           GHC.Generics
import           GHC.TypeLits
import           GHC.TypeLits.Extra
import qualified Torch as A
import           Torch.DType as D
import qualified Torch.Device as D
import           Torch.HList
import           Torch.NN (Randomizable)
import           Torch.Typed
import           Torch.Typed.Factories (RandDTypeIsValid, full)
import           Torch.Typed.Functional (relu, maxPool2d, adaptiveAvgPool2d, cat)
import           Torch.Typed.NN (Dropout, Conv2d, Conv2dSpec(..), Linear, LinearSpec(..), linear, conv2d)
import           Torch.Typed.Tensor (KnownDevice, KnownDType, Tensor, reshape)



data VggSpec (dtype :: D.DType) (device :: (D.DeviceType, Nat))
  = VggSpec deriving (Show, Eq)

  
data Vgg16 (dtype :: D.DType) device = Vgg16 {
  c0  :: Conv2d 3 64 3 3 dtype device,
  c1  :: Conv2d 64 64 3 3 dtype device,
  c2  :: Conv2d 64 128 3 3 dtype device,
  c3  :: Conv2d 128 128 3 3 dtype device,
  c4  :: Conv2d 128 256 3 3 dtype device,
  c5  :: Conv2d 256 256 3 3 dtype device,
  c6  :: Conv2d 256 256 3 3 dtype device,
  c7  :: Conv2d 256 512 3 3 dtype device,
  c8  :: Conv2d 512 512 3 3 dtype device,
  c9  :: Conv2d 512 512 3 3 dtype device,
  c10 :: Conv2d 512 512 3 3 dtype device,
  c11 :: Conv2d 512 512 3 3 dtype device,
  c12 :: Conv2d 512 512 3 3 dtype device,
  l1  :: Linear (512*7*7) 4096 dtype device,
  -- d1  :: Dropout,
  l2  :: Linear 4096 4096 dtype device,
  -- d2  :: Dropout,
  l3  :: Linear 4096 1000 dtype device
  -- d3 :: Dropout
  } deriving (Show, Generic)

instance ( KnownDType dtype
         , KnownDevice device
         , RandDTypeIsValid device dtype
         )
          => Randomizable (VggSpec dtype device) (Vgg16 dtype device) where
  sample VggSpec =
    Vgg16
    <$> A.sample (Conv2dSpec @3 @64 @3 @3)
    <*> A.sample (Conv2dSpec @64 @64 @3 @3)
    <*> A.sample (Conv2dSpec @64 @128 @3 @3)
    <*> A.sample (Conv2dSpec @128 @128 @3 @3)
    <*> A.sample (Conv2dSpec @128 @256 @3 @3)
    <*> A.sample (Conv2dSpec @256 @256 @3 @3)
    <*> A.sample (Conv2dSpec @256 @256 @3 @3)
    <*> A.sample (Conv2dSpec @256 @512 @3 @3)
    <*> A.sample (Conv2dSpec @512 @512 @3 @3)
    <*> A.sample (Conv2dSpec @512 @512 @3 @3)
    <*> A.sample (Conv2dSpec @512 @512 @3 @3)
    <*> A.sample (Conv2dSpec @512 @512 @3 @3)
    <*> A.sample (Conv2dSpec @512 @512 @3 @3)
    <*> A.sample (LinearSpec @(512*7*7) @4096)
    <*> A.sample (LinearSpec @4096 @4096)
    <*> A.sample (LinearSpec @4096 @1000)

type Stride = '(1,1)
type Pad = '(1,1)
type PoolStride = '(2,2)
type PoolKernel = '(2,2)

  -- TODO : add a nofinal
vgg16Forward :: forall batchSize dv dt . _ => Vgg16 dt dv -> Tensor dv dt '[batchSize, 3, 224, 224] -> Tensor dv dt '[batchSize, 1000]
vgg16Forward v@Vgg16{..} = linear l3 . relu . vgg16ForwardNoFinal v

vgg16ForwardNoFinal  :: forall batchSize dv dt . _ => Vgg16 dt dv -> Tensor dv dt '[batchSize, 3, 224, 224] -> Tensor dv dt '[batchSize, 4096]
vgg16ForwardNoFinal Vgg16{..} =
  linear l2 . 
  relu . 
  linear l1 . 
  reshape @'[batchSize, 512 * 7 * 7] .
  adaptiveAvgPool2d @'(7,7) . 
  maxPool2d @PoolKernel @PoolStride @Pad .
  conv2dRelu c12 .
  conv2dRelu c11 .
  conv2dRelu c10 .
  maxPool2d @PoolKernel @PoolStride @Pad .
  conv2dRelu c9 .
  conv2dRelu c8 .
  conv2dRelu c7 .
  maxPool2d @PoolKernel @PoolStride @Pad .
  conv2dRelu c6 .
  conv2dRelu c5 .
  conv2dRelu c4 .
  maxPool2d @PoolKernel @PoolStride @Pad .
  conv2dRelu c3 .
  conv2dRelu c2 .
  maxPool2d @PoolKernel @PoolStride @Pad .
  conv2dRelu c1 .
  conv2dRelu c0

  where conv2dRelu conv = relu . conv2d @Stride @Pad conv
  
normalize :: forall dv dt batchSize .  _ => Tensor dv dt [batchSize, 3, 224, 224] -> Tensor dv dt [batchSize, 3, 224, 224] 
normalize inp = ((inp / maxVal) - mean) / std 
  where
    maxVal = full @'[batchSize, 3, 224, 224] @dt @dv (255 :: Float)
    mean =
          cat @1
          $  full @'[batchSize, 1, 224, 224] @dt @dv (0.485 :: Float)
          :. full @'[batchSize, 1, 224, 224] @dt @dv (0.456 :: Float)
          :. full @'[batchSize, 1, 224, 224] @dt @dv (0.406 :: Float) 
          :. HNil
    std =
          cat @1
          $  full @'[batchSize, 1, 224, 224] @dt @dv (0.229 :: Float)
          :. full @'[batchSize, 1, 224, 224] @dt @dv (0.224 :: Float)
          :. full @'[batchSize, 1, 224, 224] @dt @dv (0.225 :: Float) 
          :. HNil
          
