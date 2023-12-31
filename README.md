# EmotionSpaces

    How consistent are the various visual emotion dataset annotations, and the theoritical emotion spaces?

----


### experiments




### datasets

| dataset | n_samples | annotations | comment |
| :-: | :-: | :-: | :-: |
|   [Abstract](https://www.imageemotion.org/testImages_abstract.zip)                      | 280/228 | Mikels 8-dim prob/clf         | prob =(argmax w/o tie)=> clf |
|   [ArtPhoto](https://www.imageemotion.org/testImages_artphoto.zip)                      | 806     | Mikels 8-dim clf              |  |
|   [Emotion6](http://chenlab.ece.cornell.edu/people/kuanchuan/publications/Emotion6.zip) | 1980    | Ekman+neutral 7-dim prob + VA |  |
|      [GAPED](https://www.unige.ch/cisa/index.php/download_file/view/288/296/)           | 730     | VA                            | 6 specific object domains, same-sized |
|  [Twitter I](https://1drv.ms/u/s!AqDZbp_iImWrhqI8k_S2uLqab_urdg?e=zwbdPG)               | 1269    | 2-dim prob                    |  |
|         [FI](https://1drv.ms/u/s!AqDZbp_iImWrhppifntgxRuw_6o2Ww?e=u2Tv7I)               | 23185   | Mikels 8-dim clf              | contain invalid samples (banned pictures) |
| [EmoSet-118K](https://www.dropbox.com/scl/fi/myue506itjfc06m7svdw6/EmoSet-118K.zip?rlkey=7f3oyjkr6zyndf0gau7t140rv&dl=0) | 118k | Mikels 8-dim + bright/colorful clf |  |

Categorical Emotion States ([ref](https://zhuanlan.zhihu.com/p/617187076)):

```
Ekman 6-dim: anger, disgust, fear, joy, sadness, surprise
=> https://www.paulekman.com/universal-emotions/

Mikels 8-dim: amusement, anger, awe, contentment, disgust, excitement, fear, sadness

Plutchik Wheel of Emotions: 
=> https://positivepsychology.com/emotion-wheel
=> https://www.jstor.org/stable/27857503?seq=1
```

### references

- surveys & essays
  - 情感计算与理解研究发展概述: [https://zhuanlan.zhihu.com/p/537984722](https://zhuanlan.zhihu.com/p/537984722)
  - Emotion Recognition from Multiple Modalities: [https://zhuanlan.zhihu.com/p/617187076](https://zhuanlan.zhihu.com/p/617187076)
  - Label Distribution Learning: [https://arxiv.org/abs/1408.6027](https://arxiv.org/abs/1408.6027)
- dataset
  - Image-Emotion-Datasets: [https://github.com/haoyev5/Image-Emotion-Datasets](https://github.com/haoyev5/Image-Emotion-Datasets)
  - Abstract & ArtPhoto: [https://www.imageemotion.org/](https://www.imageemotion.org/)
  - Emotion6: [http://chenlab.ece.cornell.edu/downloads.html](http://chenlab.ece.cornell.edu/downloads.html)
  - GAPED (see `The Geneva Affective PicturE Database (GAPED)`): [https://www.unige.ch/cisa/research/materials-and-online-research/research-material/](https://www.unige.ch/cisa/research/materials-and-online-research/research-material/)
  - Twitter I (see `Sentiment Analysis - PCNN Twitter Dataset`): [https://qzyou.github.io/](https://qzyou.github.io/)
  - FI (see `Emotion Analysis - Emotion Dataset`): [https://qzyou.github.io/](https://qzyou.github.io/)
  - UnBiasedEmo & Emotion-6: [https://rpand002.github.io/emotion.html](https://rpand002.github.io/emotion.html)
  - EmoSet: [https://github.com/JingyuanYY/EmoSet](https://github.com/JingyuanYY/EmoSet)

----
by Armit
2023/12/11 
