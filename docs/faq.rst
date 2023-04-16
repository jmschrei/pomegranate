.. _faq:

FAQ
===

**Can I create a usable model if I already know the parameters and just want to do inference**

Yes! Each model allows you to either pass in the parameters, or to leave it uninitialized and fit it directly to data. If you pass in your own parameters you can do inference by calling methods like ``log_probability`` and ``predict``.

**If I have an initial/pretrained model, can I fine-tune it using pomegranate?**

Yes! In the same way that you could just do inference after giving it parameters, you can fine-tune those parameters using the built-in fitting functions. You may want to modify the inertia or freeze some of the parameters for fine-tuning.

**If I have an initial/pretrained model, can I freeze some parameters and fine-tune the remainder?**

Yes! Do the same as above, but pass in ``frozen=True`` for the model components that you would like to remain frozen.

**How do I learn a model directly from data?**

pomegranate v1.0.0 follows the scikit-learn API in the sense that you pass all hyperparameters into the initialization and then fit the parameters using the ``fit`` function. All models allow you to use a signature similar to ``NormalDistribution().fit(X)``. Some models allow you to leave the initialization blank, but most models require at least one parameter, e.g. mixture models requires specifying the distributions and Markov chains require specifying the order. Other optional hyperparameters can be provided to alter the fitting process. the initialization is empty (or requires a few parameters, e.g. Markov chains setting the order. 

**My data set has missing values. Can I use pomegranate?**

Yes! Almost all algorithms in pomegranate can operate on incomplete data sets. All you need to do is pass in a ``torch.masked.MaskedTensor``, where the missing values are masked out (have a value of ``False``), in place of a normal tensor. 

**How can I use out-of-core learning in pomegranate?**

Once a model has been initialized the ``summarize`` method can be used on arbitrarily sized chunks of the data to reduce them into their sufficient statistics. These sufficient statistics are additive, meaning that if they are calculated for all chunks of a dataset and then added together they can yield exact updates. Once all chunks have been summarized then ``from_summaries`` is called to update the parameters of the model based on these added sufficient statistics. Out-of-core computing is supported by allowing the user to load up chunks of data from memory, summarize it, discard it, and move on to the next chunk.

**Does pomegranate support parallelization?**

Yes! Because pomegranate v1.0.0 is written in PyTorch which is natively multithreaded, all algorithms will use the available threads. See PyTorch documentation for controlling the number of threads to use.

**Does pomegranate support GPUs?**

Yes! Again, because pomegranate v1.0.0 is written in PyTorch, every algorithm has GPU support. The speed increase scales with the complexity of the algorithm, with simple probability distributions having approximately a ~2-3x speedup whereas the forward-backward algorithm for hidden Markov models can be up to ~5-10x faster by using a GPU.

**Does pomegranate support distributed computing?**

Currently pomegranate is not set up for a distributed environment, though the pieces are currently there to make this possible.

**How can I cite pomegranate?**

The research paper that presents pomegranate is:

*Schreiber, J. (2018). Pomegranate: fast and flexible probabilistic modeling in python. Journal of Machine Learning Research, 18(164), 1-6.*

which can be downloaded from `JML`_ or from `arXiv`_.

 .. _jml: http://www.jmlr.org/papers/volume18/17-636/17-636.pdf
 .. _arxiv: https://arxiv.org/abs/1711.00137

The paper can be cited as:
::

	@article{schreiber2018pomegranate,
		  title={Pomegranate: fast and flexible probabilistic modeling in python},
		  author={Schreiber, Jacob},
		  journal={Journal of Machine Learning Research},
		  volume={18},
		  number={164},
		  pages={1--6},
		  year={2018}
		}

Alternatively, the GitHub repository can be cited as:
::

	@misc{Schreiber2016,
		author = {Jacob Schreiber},
		title = {pomegranate},
		year = {2016},
		publisher = {GitHub},
		journal = {GitHub repository},
		howpublished = {\url{https://github.com/jmschrei/pomegranate}},
		commit = {enter commit that you used}
	}
