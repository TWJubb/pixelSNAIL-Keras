# pixelCNN

Experiments with pixelCNN family of generative models; including a Keras implementation of the original pixelSNAIL code (which was previously TensorFlow 1.0.

`/pixelsnail_tf1` : Refactored version of the original pixelSNAIL code, for a single GPU and working (but difficult to read and debug)

`/pixelsnail_keras` : Ported the original version into Keras; instability in code remains an issue.

## Installing

So far this needs to be used with a python environment that has been set up with the following (version numbers in brackets are the ones which work for me)

```
- tensorflow-gpu (2.1.0)
- tensorflow addons (0.9.1)
- keras (2.0.6)
```


### Instability Issues

I have found some instability with the training; occasionally it will juust produce NaN values in the weights; this is caused by the causal attention block.
