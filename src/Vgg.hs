module Vgg where

import Torch
import GHC.Generics (Generic)

-- orphan instance add to hasktorch itself
instance Parameterized [Conv2d] where

data VggSpec = VggSpec {
  convLayersSpec :: [Conv2dSpec],
  linearLayersSpec :: [LinearSpec]
  } 

data Vgg = Vgg {
  convLayers :: [Conv2d],
  linearLayers :: [Linear]
  } deriving (Generic, Parameterized)

vggLinearLayersSpec = [ LinearSpec (512 * 7 * 7) 4096,
                        LinearSpec 4096 4096,
                        LinearSpec 4096 1000
                      ]

vgg11Spec = VggSpec
  {
    convLayersSpec = [ Conv2dSpec 3 64 3 3,
                       Conv2dSpec 64 128 3 3 , 
                       Conv2dSpec 128 256 3 3 , 
                       Conv2dSpec 256 256 3 3 , 
                       Conv2dSpec 256 256 3 3 , 
                       Conv2dSpec 256 512 3 3 
                     ]
                     ++ replicate 4 (Conv2dSpec 512 512 3 3),
    linearLayersSpec = vggLinearLayersSpec
  }
vgg13Spec = VggSpec
  { convLayersSpec = [ Conv2dSpec 3 64 3 3,
                       Conv2dSpec 64 64 3 3,
                       Conv2dSpec 64 128 3 3 , 
                       Conv2dSpec 128 128 3 3 , 
                       Conv2dSpec 128 256 3 3 , 
                       Conv2dSpec 256 256 3 3 , 
                       Conv2dSpec 256 256 3 3 , 
                       Conv2dSpec 256 512 3 3
                     ]
                     ++ replicate 4 (Conv2dSpec 512 512 3 3),
    
    linearLayersSpec = vggLinearLayersSpec
    }

vgg16Spec = VggSpec
  { convLayersSpec = [ Conv2dSpec 3 64 3 3 , 
                       Conv2dSpec 64 64 3 3 , 
                       Conv2dSpec 64 128 3 3 ,
                       Conv2dSpec 128 128 3 3 , 
                       Conv2dSpec 128 256 3 3 , 
                       Conv2dSpec 256 256 3 3 , 
                       Conv2dSpec 256 256 3 3 , 
                       Conv2dSpec 256 512 3 3 
                      ] ++ replicate 5 (Conv2dSpec 512 512 3 3), 

    linearLayersSpec = vggLinearLayersSpec 
  }

vgg19Spec = VggSpec
  { convLayersSpec = [ Conv2dSpec 3 64 3 3
                     , Conv2dSpec 64 64 3 3
                     , Conv2dSpec 64 128 3 3
                     , Conv2dSpec 128 128 3 3
                     , Conv2dSpec 128 256 3 3 
                     ]
                     ++ replicate 3 (Conv2dSpec 256 256 3 3)  
                     ++ [Conv2dSpec 256 512 3 3]
                     ++ replicate 8 (Conv2dSpec 512 512 3 3),
    linearLayersSpec = vggLinearLayersSpec 
  }

instance Randomizable VggSpec Vgg where
  sample VggSpec{..} = do 
    convLayers <- sequenceA $ sample <$> convLayersSpec
    linearLayers <- sequenceA $ sample <$> linearLayersSpec
    pure $ Vgg convLayers linearLayers
  
vgg11 :: Vgg -> Tensor -> Tensor 
vgg11 weights = vggForward weights [0,1,3]

vgg13 :: Vgg -> Tensor -> Tensor 
vgg13 weights = vggForward weights [2,3,5]

vgg16 :: Vgg -> Tensor -> Tensor 
vgg16 weights = vggForward weights [1,3,6,9]

vgg19 :: Vgg -> Tensor -> Tensor 
vgg19 weights = vggForward weights [1,3,7,11]

vgg11NoFinal :: Vgg -> Tensor -> Tensor 
vgg11NoFinal weights = vggForwardNoFinal weights [0,1,3]

vgg13NoFinal :: Vgg -> Tensor -> Tensor 
vgg13NoFinal weights = vggForwardNoFinal weights [2,3,5]

vgg16NoFinal :: Vgg -> Tensor -> Tensor 
vgg16NoFinal weights = vggForwardNoFinal weights [1,3,6,9]

vgg19NoFinal :: Vgg -> Tensor -> Tensor 
vgg19NoFinal weights = vggForwardNoFinal weights [1,3,7,11]

-- not safe functions but it should be internal 
-- and we can guarantee our head and tail calls won't error at runtime
-- when we use these functions
vggForward v@Vgg{..} poolingLayerIx = linear (head linearLayers) . vggForwardNoFinal v poolingLayerIx

vggForwardNoFinal :: Vgg -> [Int] -> Tensor -> Tensor
vggForwardNoFinal Vgg{..} poolingLayerIx  input = foldl (\inp l -> relu $ linear l inp)  flattenedConv (init linearLayers) 
  where
    flattenedConv =
      flatten (Dim 1) (Dim (-1))
      $ adaptiveAvgPool2d (7,7)
      $ maxPool2d poolKernel poolStride noPad (1,1) False
      $ foldConv input

    foldConv tensor = foldl
      (\input (ix, layer)  -> if ix `elem` poolingLayerIx
                      then maxPool2d poolKernel poolStride noPad (1,1) False $ conv2dRelu layer input
                      else conv2dRelu layer input)
      tensor (zip [0..] convLayers)

    conv2dRelu layer = relu . conv2dForward layer noStride pad 
    poolStride = (2,2)
    noStride = (1,1)
    noPad = (0,0)
    pad = (1,1)
    poolKernel = (2,2)
    

  
normalize :: Int -> Tensor -> Tensor
normalize batchSize img = (img / asTensor (255 :: Float) - mean) / std 
  where
    mean = cat (Dim 1) $  (\x -> full' [batchSize,1,224,224] x)  <$> [0.485 :: Float, 0.456, 0.406]
    std = cat (Dim 1) $  (\x -> full' [batchSize,1,224,224] x) <$> [0.229 :: Float, 0.224, 0.225]
