# from pandas import read_csv
# from pandas.plotting import scatter_matrix
# from matplotlib import pyplot
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import StratifiedKFold
# from sklearn.metrics import classification_report
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import accuracy_score
# from sklearn.linear_model import LogisticRegression
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.naive_bayes import GaussianNB
# from sklearn.svm import SVC

# url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
# names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
# dataset = read_csv(url, names=names)

# dataset.shape

# print(dataset.head(20))
# print(dataset.describe)
# print(dataset.groupby('class').size())

# dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
# pyplot.show()

# dataset.hist()
# pyplot.show()

# scatter_matrix(dataset)
# pyplot.show()

# array = dataset.values
# X = array[:,0:4]
# y = array[:,4]
# X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

# models = []
# models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
# models.append(('LDA', LinearDiscriminantAnalysis()))
# models.append(('KNN', KNeighborsClassifier()))
# models.append(('CART', DecisionTreeClassifier()))
# models.append(('NB', GaussianNB()))
# models.append(('SVM', SVC(gamma='auto')))
# # evaluate each model in turn
# results = []
# names = []
# for name, model in models:
#  kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
#  cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
#  results.append(cv_results)
#  names.append(name)
#  print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

# pyplot.boxplot(results, labels=names)
# pyplot.title('Algorithm Comparison')
# pyplot.show()

# model = SVC(gamma='auto')
# model.fit(X_train, Y_train)
# predictions = model.predict(X_validation)

# print(accuracy_score(Y_validation, predictions))
# print(confusion_matrix(Y_validation, predictions))
# print(classification_report(Y_validation, predictions))


# /////////////////////////////////////////////////////////////////////////////////////////////////////////

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sb
 
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from xgboost import XGBClassifier
# from sklearn import metrics
 
# import warnings
# warnings.filterwarnings('ignore')

# df=pd.read_csv('c:/users/wston/documents/python/pybuild/tsla.csv')
# print(df.head())


# x=df.shape
# y=df.describe
# z=df.info

# print(x)
# print(y)
# print(z)

# plt.figure(figsize=(15,5))
# plt.plot(df['Close'])
# plt.title('Tesla Close price.', fontsize=15)
# plt.ylabel('Price in dollars.')
# plt.show()

# df.head()

# df[df['Close'] == df['Adj Close']].shape

# df = df.drop(['Adj Close'], axis=1)


# df.isnull().sum()


# features = ['Open', 'High', 'Low', 'Close', 'Volume']
 
# plt.subplots(figsize=(20,10))
 
# for i, col in enumerate(features):
#   plt.subplot(2,3,i+1)
#   sb.distplot(df[col])
# plt.show()

# plt.subplots(figsize=(20,10))
# for i, col in enumerate(features):
#   plt.subplot(2,3,i+1)
#   sb.boxplot(df[col])
# plt.show()


# splitted = df['Date'].str.split('/', expand=True)
 
# df['day'] = splitted[1].astype('int')
# df['month'] = splitted[0].astype('int')
# df['year'] = splitted[2].astype('int')
 
# df.head()

# df['is_quarter_end'] = np.where(df['month']%3==0,1,0)
# df.head()


# data_grouped = df.groupby('year').mean()
# plt.subplots(figsize=(20,10))
 
# for i, col in enumerate(['Open', 'High', 'Low', 'Close']):
#   plt.subplot(2,2,i+1)
#   data_grouped[col].plot.bar()
# plt.show()


# df.groupby('is_quarter_end').mean()


# df['open-close']  = df['Open'] - df['Close']
# df['low-high']  = df['Low'] - df['High']
# df['target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)


# plt.pie(df['target'].value_counts().values,
#         labels=[0, 1], autopct='%1.1f%%')
# plt.show()


# plt.figure(figsize=(10, 10))
 
# # As our concern is with the highly
# # correlated features only so, we will visualize
# # our heatmap as per that criteria only.
# sb.heatmap(df.corr() > 0.9, annot=True, cbar=False)
# plt.show()


# features = df[['open-close', 'low-high', 'is_quarter_end']]
# target = df['target']
 
# scaler = StandardScaler()
# features = scaler.fit_transform(features)
 
# X_train, X_valid, Y_train, Y_valid = train_test_split(
#     features, target, test_size=0.1, random_state=2022)
# print(X_train.shape, X_valid.shape)



# models = [LogisticRegression(), SVC(
#   kernel='poly', probability=True), XGBClassifier()]
 
# for i in range(3):
#   models[i].fit(X_train, Y_train)
 
#   print(f'{models[i]} : ')
#   print('Training Accuracy : ', metrics.roc_auc_score(
#     Y_train, models[i].predict_proba(X_train)[:,1]))
#   print('Validation Accuracy : ', metrics.roc_auc_score(
#     Y_valid, models[i].predict_proba(X_valid)[:,1]))
#   print()


  
# metrics.plot_confusion_matrix(models[0], X_valid, Y_valid)
# plt.show()sklearn.tree import DecisionTreeClassifier
# # from sklearn.neighbors import KNeighborsClassifier

# ////////////////////////////////////////////////////////////////////////////////////////////////////////////

# from sklearn import linear_model
# reg = linear_model.LinearRegression()

# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn import datasets, linear_model
# from sklearn.metrics import mean_squared_error, r2_score

# # Load the diabetes dataset
# diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

# # Use only one feature
# diabetes_X = diabetes_X[:, np.newaxis, 2]

# # Split the data into training/testing sets
# diabetes_X_train = diabetes_X[:-20]
# diabetes_X_test = diabetes_X[-20:]

# # Split the targets into training/testing sets
# diabetes_y_train = diabetes_y[:-20]
# diabetes_y_test = diabetes_y[-20:]

# # Create linear regression object
# regr = linear_model.LinearRegression()

# # Train the model using the training sets
# regr.fit(diabetes_X_train, diabetes_y_train)

# # Make predictions using the testing set
# diabetes_y_pred = regr.predict(diabetes_X_test)

# # The coefficients
# print("Coefficients: \n", regr.coef_)
# # The mean squared error
# print("Mean squared error: %.2f" % mean_squared_error(diabetes_y_test, diabetes_y_pred))
# # The coefficient of determination: 1 is perfect prediction
# print("Coefficient of determination: %.2f" % r2_score(diabetes_y_test, diabetes_y_pred))

# # Plot outputs
# plt.scatter(diabetes_X_test, diabetes_y_test, color="black")
# plt.plot(diabetes_X_test, diabetes_y_pred, color="blue", linewidth=3)

# plt.xticks(())
# plt.yticks(())
# plt.show()

# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////

# import sys
# import os
# from os.path import join
# import platform
# import shutil

# from setuptools import Command, Extension, setup
# from setuptools.command.build_ext import build_ext

# import traceback
# import importlib

# try:
#     import builtins
# except ImportError:
#     # Python 2 compat: just to be able to declare that Python >=3.8 is needed.
#     import __builtin__ as builtins

# # This is a bit (!) hackish: we are setting a global variable so that the main
# # sklearn __init__ can detect if it is being loaded by the setup routine, to
# # avoid attempting to load components that aren't built yet.
# # TODO: can this be simplified or removed since the switch to setuptools
# # away from numpy.distutils?
# builtins.__SKLEARN_SETUP__ = True


# DISTNAME = "scikit-learn"
# DESCRIPTION = "A set of python modules for machine learning and data mining"
# with open("README.rst") as f:
#     LONG_DESCRIPTION = f.read()
# MAINTAINER = "Andreas Mueller"
# MAINTAINER_EMAIL = "amueller@ais.uni-bonn.de"
# URL = "http://scikit-learn.org"
# DOWNLOAD_URL = "https://pypi.org/project/scikit-learn/#files"
# LICENSE = "new BSD"
# PROJECT_URLS = {
#     "Bug Tracker": "https://github.com/scikit-learn/scikit-learn/issues",
#     "Documentation": "https://scikit-learn.org/stable/documentation.html",
#     "Source Code": "https://github.com/scikit-learn/scikit-learn",
# }

# # We can actually import a restricted version of sklearn that
# # does not need the compiled code
# import sklearn  # noqa
# import sklearn._min_dependencies as min_deps  # noqa
# from sklearn._build_utils import _check_cython_version  # noqa
# from sklearn.externals._packaging.version import parse as parse_version  # noqa


# VERSION = sklearn.__version__

# # See: https://numpy.org/doc/stable/reference/c-api/deprecations.html
# DEFINE_MACRO_NUMPY_C_API = (
#     "NPY_NO_DEPRECATED_API",
#     "NPY_1_7_API_VERSION",
# )

# # XXX: add new extensions to this list when they
# # are not using the old NumPy C API (i.e. version 1.7)
# # TODO: when Cython>=3.0 is used, make sure all Cython extensions
# # use the newest NumPy C API by `#defining` `NPY_NO_DEPRECATED_API` to be
# # `NPY_1_7_API_VERSION`, and remove this list.
# # See: https://github.com/cython/cython/blob/1777f13461f971d064bd1644b02d92b350e6e7d1/docs/src/userguide/migrating_to_cy30.rst#numpy-c-api # noqa
# USE_NEWEST_NUMPY_C_API = (
#     "sklearn.__check_build._check_build",
#     "sklearn._loss._loss",
#     "sklearn._isotonic",
#     "sklearn.cluster._dbscan_inner",
#     "sklearn.cluster._hierarchical_fast",
#     "sklearn.cluster._k_means_common",
#     "sklearn.cluster._k_means_lloyd",
#     "sklearn.cluster._k_means_elkan",
#     "sklearn.cluster._k_means_minibatch",
#     "sklearn.datasets._svmlight_format_fast",
#     "sklearn.decomposition._cdnmf_fast",
#     "sklearn.decomposition._online_lda_fast",
#     "sklearn.ensemble._gradient_boosting",
#     "sklearn.ensemble._hist_gradient_boosting._gradient_boosting",
#     "sklearn.ensemble._hist_gradient_boosting.histogram",
#     "sklearn.ensemble._hist_gradient_boosting.splitting",
#     "sklearn.ensemble._hist_gradient_boosting._binning",
#     "sklearn.ensemble._hist_gradient_boosting._predictor",
#     "sklearn.ensemble._hist_gradient_boosting._bitset",
#     "sklearn.ensemble._hist_gradient_boosting.common",
#     "sklearn.ensemble._hist_gradient_boosting.utils",
#     "sklearn.feature_extraction._hashing_fast",
#     "sklearn.linear_model._sag_fast",
#     "sklearn.linear_model._sgd_fast",
#     "sklearn.manifold._barnes_hut_tsne",
#     "sklearn.manifold._utils",
#     "sklearn.metrics.cluster._expected_mutual_info_fast",
#     "sklearn.metrics._pairwise_distances_reduction._datasets_pair",
#     "sklearn.metrics._pairwise_distances_reduction._middle_term_computer",
#     "sklearn.metrics._pairwise_distances_reduction._base",
#     "sklearn.metrics._pairwise_distances_reduction._argkmin",
#     "sklearn.metrics._pairwise_distances_reduction._radius_neighbors",
#     "sklearn.metrics._pairwise_fast",
#     "sklearn.neighbors._ball_tree",
#     "sklearn.neighbors._kd_tree",
#     "sklearn.neighbors._partition_nodes",
#     "sklearn.neighbors._quad_tree",
#     "sklearn.preprocessing._csr_polynomial_expansion",
#     "sklearn.svm._liblinear",
#     "sklearn.svm._libsvm",
#     "sklearn.svm._libsvm_sparse",
#     "sklearn.svm._newrand",
#     "sklearn.tree._criterion",
#     "sklearn.tree._splitter",
#     "sklearn.tree._tree",
#     "sklearn.tree._utils",
#     "sklearn.utils._cython_blas",
#     "sklearn.utils._fast_dict",
#     "sklearn.utils._heap",
#     "sklearn.utils._isfinite",
#     "sklearn.utils._logistic_sigmoid",
#     "sklearn.utils._openmp_helpers",
#     "sklearn.utils._random",
#     "sklearn.utils._seq_dataset",
#     "sklearn.utils._sorting",
#     "sklearn.utils._typedefs",
#     "sklearn.utils._vector_sentinel",
#     "sklearn.utils._weight_vector",
#     "sklearn.utils.murmurhash",
# )


# # Custom clean command to remove build artifacts


# class CleanCommand(Command):
#     description = "Remove build artifacts from the source tree"

#     user_options = []

#     def initialize_options(self):
#         pass

#     def finalize_options(self):
#         pass

#     def run(self):
#         # Remove c files if we are not within a sdist package
#         cwd = os.path.abspath(os.path.dirname(__file__))
#         remove_c_files = not os.path.exists(os.path.join(cwd, "PKG-INFO"))
#         if remove_c_files:
#             print("Will remove generated .c files")
#         if os.path.exists("build"):
#             shutil.rmtree("build")
#         for dirpath, dirnames, filenames in os.walk("sklearn"):
#             for filename in filenames:
#                 if any(
#                     filename.endswith(suffix)
#                     for suffix in (".so", ".pyd", ".dll", ".pyc")
#                 ):
#                     os.unlink(os.path.join(dirpath, filename))
#                     continue
#                 extension = os.path.splitext(filename)[1]
#                 if remove_c_files and extension in [".c", ".cpp"]:
#                     pyx_file = str.replace(filename, extension, ".pyx")
#                     if os.path.exists(os.path.join(dirpath, pyx_file)):
#                         os.unlink(os.path.join(dirpath, filename))
#             for dirname in dirnames:
#                 if dirname == "__pycache__":
#                     shutil.rmtree(os.path.join(dirpath, dirname))


# # Custom build_ext command to set OpenMP compile flags depending on os and
# # compiler. Also makes it possible to set the parallelism level via
# # and environment variable (useful for the wheel building CI).
# # build_ext has to be imported after setuptools


# class build_ext_subclass(build_ext):
#     def finalize_options(self):
#         build_ext.finalize_options(self)
#         if self.parallel is None:
#             # Do not override self.parallel if already defined by
#             # command-line flag (--parallel or -j)

#             parallel = os.environ.get("SKLEARN_BUILD_PARALLEL")
#             if parallel:
#                 self.parallel = int(parallel)
#         if self.parallel:
#             print("setting parallel=%d " % self.parallel)

#     def build_extensions(self):
#         from sklearn._build_utils.openmp_helpers import get_openmp_flag

#         for ext in self.extensions:
#             if ext.name in USE_NEWEST_NUMPY_C_API:
#                 print(f"Using newest NumPy C API for extension {ext.name}")
#                 ext.define_macros.append(DEFINE_MACRO_NUMPY_C_API)
#             else:
#                 print(f"Using old NumPy C API (version 1.7) for extension {ext.name}")

#         if sklearn._OPENMP_SUPPORTED:
#             openmp_flag = get_openmp_flag(self.compiler)

#             for e in self.extensions:
#                 e.extra_compile_args += openmp_flag
#                 e.extra_link_args += openmp_flag

#         build_ext.build_extensions(self)

#     def run(self):
#         # Specifying `build_clib` allows running `python setup.py develop`
#         # fully from a fresh clone.
#         self.run_command("build_clib")
#         build_ext.run(self)


# cmdclass = {
#     "clean": CleanCommand,
#     "build_ext": build_ext_subclass,
# }


# def check_package_status(package, min_version):
#     """
#     Returns a dictionary containing a boolean specifying whether given package
#     is up-to-date, along with the version string (empty string if
#     not installed).
#     """
#     package_status = {}
#     try:
#         module = importlib.import_module(package)
#         package_version = module.__version__
#         package_status["up_to_date"] = parse_version(package_version) >= parse_version(
#             min_version
#         )
#         package_status["version"] = package_version
#     except ImportError:
#         traceback.print_exc()
#         package_status["up_to_date"] = False
#         package_status["version"] = ""

#     req_str = "scikit-learn requires {} >= {}.\n".format(package, min_version)

#     instructions = (
#         "Installation instructions are available on the "
#         "scikit-learn website: "
#         "http://scikit-learn.org/stable/install.html\n"
#     )

#     if package_status["up_to_date"] is False:
#         if package_status["version"]:
#             raise ImportError(
#                 "Your installation of {} {} is out-of-date.\n{}{}".format(
#                     package, package_status["version"], req_str, instructions
#                 )
#             )
#         else:
#             raise ImportError(
#                 "{} is not installed.\n{}{}".format(package, req_str, instructions)
#             )


# extension_config = {
#     "__check_build": [
#         {"sources": ["_check_build.pyx"]},
#     ],
#     "": [
#         {"sources": ["_isotonic.pyx"], "include_np": True},
#     ],
#     "_loss": [
#         {"sources": ["_loss.pyx.tp"]},
#     ],
#     "cluster": [
#         {"sources": ["_dbscan_inner.pyx"], "language": "c++", "include_np": True},
#         {"sources": ["_hierarchical_fast.pyx"], "language": "c++", "include_np": True},
#         {"sources": ["_k_means_common.pyx"], "include_np": True},
#         {"sources": ["_k_means_lloyd.pyx"], "include_np": True},
#         {"sources": ["_k_means_elkan.pyx"], "include_np": True},
#         {"sources": ["_k_means_minibatch.pyx"], "include_np": True},
#     ],
#     "datasets": [
#         {
#             "sources": ["_svmlight_format_fast.pyx"],
#             "include_np": True,
#             "compile_for_pypy": False,
#         }
#     ],
#     "decomposition": [
#         {"sources": ["_online_lda_fast.pyx"], "include_np": True},
#         {"sources": ["_cdnmf_fast.pyx"], "include_np": True},
#     ],
#     "ensemble": [
#         {"sources": ["_gradient_boosting.pyx"], "include_np": True},
#     ],
#     "ensemble._hist_gradient_boosting": [
#         {"sources": ["_gradient_boosting.pyx"], "include_np": True},
#         {"sources": ["histogram.pyx"], "include_np": True},
#         {"sources": ["splitting.pyx"], "include_np": True},
#         {"sources": ["_binning.pyx"], "include_np": True},
#         {"sources": ["_predictor.pyx"], "include_np": True},
#         {"sources": ["_bitset.pyx"], "include_np": True},
#         {"sources": ["common.pyx"], "include_np": True},
#         {"sources": ["utils.pyx"], "include_np": True},
#     ],
#     "feature_extraction": [
#         {"sources": ["_hashing_fast.pyx"], "language": "c++", "include_np": True},
#     ],
#     "linear_model": [
#         {"sources": ["_cd_fast.pyx"], "include_np": True},
#         {"sources": ["_sgd_fast.pyx"], "include_np": True},
#         {"sources": ["_sag_fast.pyx.tp"], "include_np": True},
#     ],
#     "manifold": [
#         {"sources": ["_utils.pyx"], "include_np": True},
#         {"sources": ["_barnes_hut_tsne.pyx"], "include_np": True},
#     ],
#     "metrics": [
#         {"sources": ["_pairwise_fast.pyx"], "include_np": True},
#         {
#             "sources": ["_dist_metrics.pyx.tp", "_dist_metrics.pxd.tp"],
#             "include_np": True,
#         },
#     ],
#     "metrics.cluster": [
#         {"sources": ["_expected_mutual_info_fast.pyx"], "include_np": True},
#     ],
#     "metrics._pairwise_distances_reduction": [
#         {
#             "sources": ["_datasets_pair.pyx.tp", "_datasets_pair.pxd.tp"],
#             "language": "c++",
#             "include_np": True,
#             "extra_compile_args": ["-std=c++11"],
#         },
#         {
#             "sources": ["_middle_term_computer.pyx.tp", "_middle_term_computer.pxd.tp"],
#             "language": "c++",
#             "include_np": True,
#             "extra_compile_args": ["-std=c++11"],
#         },
#         {
#             "sources": ["_base.pyx.tp", "_base.pxd.tp"],
#             "language": "c++",
#             "include_np": True,
#             "extra_compile_args": ["-std=c++11"],
#         },
#         {
#             "sources": ["_argkmin.pyx.tp", "_argkmin.pxd.tp"],
#             "language": "c++",
#             "include_np": True,
#             "extra_compile_args": ["-std=c++11"],
#         },
#         {
#             "sources": ["_radius_neighbors.pyx.tp", "_radius_neighbors.pxd.tp"],
#             "language": "c++",
#             "include_np": True,
#             "extra_compile_args": ["-std=c++11"],
#         },
#     ],
#     "preprocessing": [
#         {"sources": ["_csr_polynomial_expansion.pyx"], "include_np": True},
#     ],
#     "neighbors": [
#         {"sources": ["_ball_tree.pyx"], "include_np": True},
#         {"sources": ["_kd_tree.pyx"], "include_np": True},
#         {"sources": ["_partition_nodes.pyx"], "language": "c++", "include_np": True},
#         {"sources": ["_quad_tree.pyx"], "include_np": True},
#     ],
#     "svm": [
#         {
#             "sources": ["_newrand.pyx"],
#             "include_np": True,
#             "include_dirs": [join("src", "newrand")],
#             "language": "c++",
#             # Use C++11 random number generator fix
#             "extra_compile_args": ["-std=c++11"],
#         },
#         {
#             "sources": ["_libsvm.pyx"],
#             "depends": [
#                 join("src", "libsvm", "libsvm_helper.c"),
#                 join("src", "libsvm", "libsvm_template.cpp"),
#                 join("src", "libsvm", "svm.cpp"),
#                 join("src", "libsvm", "svm.h"),
#                 join("src", "newrand", "newrand.h"),
#             ],
#             "include_dirs": [
#                 join("src", "libsvm"),
#                 join("src", "newrand"),
#             ],
#             "libraries": ["libsvm-skl"],
#             "extra_link_args": ["-lstdc++"],
#             "include_np": True,
#         },
#         {
#             "sources": ["_liblinear.pyx"],
#             "libraries": ["liblinear-skl"],
#             "include_dirs": [
#                 join("src", "liblinear"),
#                 join("src", "newrand"),
#                 join("..", "utils"),
#             ],
#             "include_np": True,
#             "depends": [
#                 join("src", "liblinear", "tron.h"),
#                 join("src", "liblinear", "linear.h"),
#                 join("src", "liblinear", "liblinear_helper.c"),
#                 join("src", "newrand", "newrand.h"),
#             ],
#             "extra_link_args": ["-lstdc++"],
#         },
#         {
#             "sources": ["_libsvm_sparse.pyx"],
#             "libraries": ["libsvm-skl"],
#             "include_dirs": [
#                 join("src", "libsvm"),
#                 join("src", "newrand"),
#             ],
#             "include_np": True,
#             "depends": [
#                 join("src", "libsvm", "svm.h"),
#                 join("src", "newrand", "newrand.h"),
#                 join("src", "libsvm", "libsvm_sparse_helper.c"),
#             ],
#             "extra_link_args": ["-lstdc++"],
#         },
#     ],
#     "tree": [
#         {
#             "sources": ["_tree.pyx"],
#             "language": "c++",
#             "include_np": True,
#             "optimization_level": "O3",
#         },
#         {"sources": ["_splitter.pyx"], "include_np": True, "optimization_level": "O3"},
#         {"sources": ["_criterion.pyx"], "include_np": True, "optimization_level": "O3"},
#         {"sources": ["_utils.pyx"], "include_np": True, "optimization_level": "O3"},
#     ],
#     "utils": [
#         {"sources": ["sparsefuncs_fast.pyx"], "include_np": True},
#         {"sources": ["_cython_blas.pyx"]},
#         {"sources": ["arrayfuncs.pyx"], "include_np": True},
#         {
#             "sources": ["murmurhash.pyx", join("src", "MurmurHash3.cpp")],
#             "include_dirs": ["src"],
#             "include_np": True,
#         },
#         {"sources": ["_fast_dict.pyx"], "language": "c++", "include_np": True},
#         {"sources": ["_fast_dict.pyx"], "language": "c++", "include_np": True},
#         {"sources": ["_openmp_helpers.pyx"]},
#         {"sources": ["_seq_dataset.pyx.tp", "_seq_dataset.pxd.tp"], "include_np": True},
#         {
#             "sources": ["_weight_vector.pyx.tp", "_weight_vector.pxd.tp"],
#             "include_np": True,
#         },
#         {"sources": ["_random.pyx"], "include_np": True},
#         {"sources": ["_logistic_sigmoid.pyx"], "include_np": True},
#         {"sources": ["_typedefs.pyx"], "include_np": True},
#         {"sources": ["_heap.pyx"], "include_np": True},
#         {"sources": ["_sorting.pyx"], "include_np": True},
#         {"sources": ["_vector_sentinel.pyx"], "language": "c++", "include_np": True},
#         {"sources": ["_isfinite.pyx"]},
#     ],
# }

# # Paths in `libraries` must be relative to the root directory because `libraries` is
# # passed directly to `setup`
# libraries = [
#     (
#         "libsvm-skl",
#         {
#             "sources": [
#                 join("sklearn", "svm", "src", "libsvm", "libsvm_template.cpp"),
#             ],
#             "depends": [
#                 join("sklearn", "svm", "src", "libsvm", "svm.cpp"),
#                 join("sklearn", "svm", "src", "libsvm", "svm.h"),
#                 join("sklearn", "svm", "src", "newrand", "newrand.h"),
#             ],
#             # Use C++11 to use the random number generator fix
#             "extra_compiler_args": ["-std=c++11"],
#             "extra_link_args": ["-lstdc++"],
#         },
#     ),
#     (
#         "liblinear-skl",
#         {
#             "sources": [
#                 join("sklearn", "svm", "src", "liblinear", "linear.cpp"),
#                 join("sklearn", "svm", "src", "liblinear", "tron.cpp"),
#             ],
#             "depends": [
#                 join("sklearn", "svm", "src", "liblinear", "linear.h"),
#                 join("sklearn", "svm", "src", "liblinear", "tron.h"),
#                 join("sklearn", "svm", "src", "newrand", "newrand.h"),
#             ],
#             # Use C++11 to use the random number generator fix
#             "extra_compiler_args": ["-std=c++11"],
#             "extra_link_args": ["-lstdc++"],
#         },
#     ),
# ]


# def configure_extension_modules():
#     # Skip cythonization as we do not want to include the generated
#     # C/C++ files in the release tarballs as they are not necessarily
#     # forward compatible with future versions of Python for instance.
#     if "sdist" in sys.argv or "--help" in sys.argv:
#         return []

#     from sklearn._build_utils import cythonize_extensions
#     from sklearn._build_utils import gen_from_templates
#     import numpy

#     is_pypy = platform.python_implementation() == "PyPy"
#     np_include = numpy.get_include()
#     default_optimization_level = "O2"

#     if os.name == "posix":
#         default_libraries = ["m"]
#     else:
#         default_libraries = []

#     default_extra_compile_args = []
#     build_with_debug_symbols = (
#         os.environ.get("SKLEARN_BUILD_ENABLE_DEBUG_SYMBOLS", "0") != "0"
#     )
#     if os.name == "posix":
#         if build_with_debug_symbols:
#             default_extra_compile_args.append("-g")
#         else:
#             # Setting -g0 will strip symbols, reducing the binary size of extensions
#             default_extra_compile_args.append("-g0")

#     cython_exts = []
#     for submodule, extensions in extension_config.items():
#         submodule_parts = submodule.split(".")
#         parent_dir = join("sklearn", *submodule_parts)
#         for extension in extensions:
#             if is_pypy and not extension.get("compile_for_pypy", True):
#                 continue

#             # Generate files with Tempita
#             tempita_sources = []
#             sources = []
#             for source in extension["sources"]:
#                 source = join(parent_dir, source)
#                 new_source_path, path_ext = os.path.splitext(source)

#                 if path_ext != ".tp":
#                     sources.append(source)
#                     continue

#                 # `source` is a Tempita file
#                 tempita_sources.append(source)

#                 # Do not include pxd files that were generated by tempita
#                 if os.path.splitext(new_source_path)[-1] == ".pxd":
#                     continue
#                 sources.append(new_source_path)

#             gen_from_templates(tempita_sources)

#             # By convention, our extensions always use the name of the first source
#             source_name = os.path.splitext(os.path.basename(sources[0]))[0]
#             if submodule:
#                 name_parts = ["sklearn", submodule, source_name]
#             else:
#                 name_parts = ["sklearn", source_name]
#             name = ".".join(name_parts)

#             # Make paths start from the root directory
#             include_dirs = [
#                 join(parent_dir, include_dir)
#                 for include_dir in extension.get("include_dirs", [])
#             ]
#             if extension.get("include_np", False):
#                 include_dirs.append(np_include)

#             depends = [
#                 join(parent_dir, depend) for depend in extension.get("depends", [])
#             ]

#             extra_compile_args = (
#                 extension.get("extra_compile_args", []) + default_extra_compile_args
#             )
#             optimization_level = extension.get(
#                 "optimization_level", default_optimization_level
#             )
#             if os.name == "posix":
#                 extra_compile_args.append(f"-{optimization_level}")
#             else:
#                 extra_compile_args.append(f"/{optimization_level}")

#             libraries_ext = extension.get("libraries", []) + default_libraries

#             new_ext = Extension(
#                 name=name,
#                 sources=sources,
#                 language=extension.get("language", None),
#                 include_dirs=include_dirs,
#                 libraries=libraries_ext,
#                 depends=depends,
#                 extra_link_args=extension.get("extra_link_args", None),
#                 extra_compile_args=extra_compile_args,
#             )
#             cython_exts.append(new_ext)

#     return cythonize_extensions(cython_exts)


# def setup_package():
#     python_requires = ">=3.8"
#     required_python_version = (3, 8)

#     metadata = dict(
#         name=DISTNAME,
#         maintainer=MAINTAINER,
#         maintainer_email=MAINTAINER_EMAIL,
#         description=DESCRIPTION,
#         license=LICENSE,
#         url=URL,
#         download_url=DOWNLOAD_URL,
#         project_urls=PROJECT_URLS,
#         version=VERSION,
#         long_description=LONG_DESCRIPTION,
#         classifiers=[
#             "Intended Audience :: Science/Research",
#             "Intended Audience :: Developers",
#             "License :: OSI Approved :: BSD License",
#             "Programming Language :: C",
#             "Programming Language :: Python",
#             "Topic :: Software Development",
#             "Topic :: Scientific/Engineering",
#             "Development Status :: 5 - Production/Stable",
#             "Operating System :: Microsoft :: Windows",
#             "Operating System :: POSIX",
#             "Operating System :: Unix",
#             "Operating System :: MacOS",
#             "Programming Language :: Python :: 3",
#             "Programming Language :: Python :: 3.8",
#             "Programming Language :: Python :: 3.9",
#             "Programming Language :: Python :: 3.10",
#             "Programming Language :: Python :: 3.11",
#             "Programming Language :: Python :: Implementation :: CPython",
#             "Programming Language :: Python :: Implementation :: PyPy",
#         ],
#         cmdclass=cmdclass,
#         python_requires=python_requires,
#         install_requires=min_deps.tag_to_packages["install"],
#         package_data={"": ["*.csv", "*.gz", "*.txt", "*.pxd", "*.rst", "*.jpg"]},
#         zip_safe=False,  # the package can run out of an .egg file
#         extras_require={
#             key: min_deps.tag_to_packages[key]
#             for key in ["examples", "docs", "tests", "benchmark"]
#         },
#     )

#     commands = [arg for arg in sys.argv[1:] if not arg.startswith("-")]
#     if not all(
#         command in ("egg_info", "dist_info", "clean", "check") for command in commands
#     ):
#         if sys.version_info < required_python_version:
#             required_version = "%d.%d" % required_python_version
#             raise RuntimeError(
#                 "Scikit-learn requires Python %s or later. The current"
#                 " Python version is %s installed in %s."
#                 % (required_version, platform.python_version(), sys.executable)
#             )

#         check_package_status("numpy", min_deps.NUMPY_MIN_VERSION)
#         check_package_status("scipy", min_deps.SCIPY_MIN_VERSION)

#         _check_cython_version()
#         metadata["ext_modules"] = configure_extension_modules()
#         metadata["libraries"] = libraries
#     setup(**metadata)


# if __name__ == "__main__":
#     setup_package()

# ////////////////////////////////////////////////////////////////////////////////////////////////////////////

# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd

# from sklearn.datasets import make_regression

# foam = pd.read_csv('c:/users/wston/documents/beerdata.csv')
# foam.head()

# X = foam[["foam", "beer"]]
# y = foam["time"].values.reshape(-1, 1)
# print(X.shape, y.shape)

# from sklearn.model_selection import train_test_split

# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# from sklearn.linear_model import LinearRegression
# model = LinearRegression()

# model.fit(X_train, y_train)
# training_score = model.score(X_train, y_train)
# testing_score = model.score(X_test, y_test)


# print(f"Training Score: {training_score}")
# print(f"Testing Score: {testing_score}")

# plt.scatter(model.predict(X_train), model.predict(X_train) - y_train, c="blue", label="Training Data")
# plt.scatter(model.predict(X_test), model.predict(X_test) - y_test, c="orange", label="Testing Data")
# plt.legend()
# plt.hlines(y=0, xmin=y.min(), xmax=y.max())
# plt.title("Residual Plot")
# plt.show()

# from mpl_toolkits.mplot3d import Axes3D
# fig = plt.figure(1, figsize=(5, 5))
# axes = Axes3D(fig, elev=20, azim=45)
# axes.scatter(foam['time'],foam['beer'],foam['foam'], c=y, cmap=plt.cm.get_cmap("Spectral"))
# plt.show()



# # Import libraries
# from mpl_toolkits import mplot3d
# import numpy as np
# import matplotlib.pyplot as plt
 
 
# # Creating dataset
# x = foam['beer']
# y = foam['foam']
# z = foam['time'] 

# # Creating figure
# fig = plt.figure(figsize = (10, 7))
# ax = plt.axes(projection ="3d")
 
# # Creating plot
# ax.scatter3D(x, y, z, color = "green")
# plt.title("simple 3D scatter plot")
 
# # show plot
# plt.show()

# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd

# from sklearn.datasets import make_regression

# X, y = make_regression(n_samples=20, n_features=1, random_state=0, noise=4, bias=100.0)
# plt.scatter(X, y)
# plt.show()

#////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

import pandas as pd 
import numpy as np

file1 = 'c:/users/wston/documents/python/pybuild/TSLA.csv'

jump = pd.read_csv(file1)
datetime_series = pd.DatetimeIndex(jump['Date'])
datetime_series
jump = jump.set_index(datetime_series)
jump
jump.to_csv('c:/users/wston/documents/stock_data.csv')

num_obs = jump[['Close']].count()
num_obs


def add_memory(s,n_days=50,mem_strength=0.1):
    ''' adds autoregressive behavior to series of data'''
    add_ewm = lambda x: (1-mem_strength)*x + mem_strength*x.ewm(n_days).mean()
    out = s.groupby(level='Date').apply(add_ewm)
    return out

# generate feature data
f01 = pd.Series(np.random.randn(3182),index=jump[['Close']].index)
f01 = add_memory(f01,10,0.1)
f02 = pd.Series(np.random.randn(3182),index=jump[['Close']].index)
f02 = add_memory(f02,10,0.1)
f03 = pd.Series(np.random.randn(3182),index=jump[['Close']].index)
f03 = add_memory(f03,10,0.1)
f04 = pd.Series(np.random.randn(3182),index=jump[['Close']].index)
f04 = f04 # no memory

features = pd.concat([f01,f02,f03,f04],axis=1)
features.to_csv('c:/users/wston/documents/jump_data.csv')

# dim_feat=features.ndim
# dim_feat

# now, create response variable such that it is related to features
# f01 becomes increasingly important, f02 becomes decreasingly important,
# f03 oscillates in importance, f04 is stationary, and finally a noise component is added

outcome =   f01 * np.linspace(0.5,1.5,3182) + \
            f02 * np.linspace(1.5,0.5,3182) + \
            f03 * pd.Series(np.sin(2*np.pi*np.linspace(0,1,3182)*2)+1,index=f03.index) + \
            f04 + \
            np.random.randn(3182) * 3 
outcome.name = 'outcome'
outcome

from sklearn.linear_model import LinearRegression
from itertools import chain

recalc_dates = features.resample('D',level='Date').mean().values[:-1]
# print('init', str(recalc_dates))

flat = list(chain.from_iterable(recalc_dates))
print(flat)
recalc_dates = pd.Series(flat)
models = pd.Series(index=recalc_dates)
for date in recalc_dates:
    X_train = features.iloc[0:3182]
    y_train = outcome.iloc[0:3182]
    model = LinearRegression()
    model.fit(X_train,y_train)
    models.loc[date] = model

## predict values walk-forward (all predictions out of sample)
begin_dates = models.index
end_dates = models.index[1:].append(pd.to_datetime(['2099-12-31']))

predictions = pd.Series(index=features.index)

for i,model in enumerate(models): #loop thru each models object in collection
    X = features.iloc[0:3182]
    p = pd.Series(model.predict(X),index=X.index)
    predictions.loc[X.index] = p