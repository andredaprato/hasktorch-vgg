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
                       Conv2dSpec 256 256 3 3 , 
                       Conv2dSpec 256 512 3 3 
                      ] ++ replicate 4 (Conv2dSpec 512 512 3 3), 

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
  
