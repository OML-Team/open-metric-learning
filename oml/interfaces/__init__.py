<<<<<<< HEAD
from oml.interfaces.criterions import ITripletLossWithMiner
from oml.interfaces.miners import (
    labels2list,
    ITripletsMiner,
    ITripletsMinerInBatch,
    TLabels,
    TTripletsIds,
    TTriplets
)
from oml.interfaces.models import (
    IExtractor, IFreezable, IPairwiseModel
)
=======
from .criterions import ITripletLossWithMiner
from .miners import labels2list, ITripletsMiner, ITripletsMinerInBatch
>>>>>>> parent of 8469696 (Simplify 'models' module imports)
