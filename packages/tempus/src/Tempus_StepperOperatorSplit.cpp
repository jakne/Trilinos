// @HEADER
// ****************************************************************************
//                Tempus: Copyright (2017) Sandia Corporation
//
// Distributed under BSD 3-clause license (See accompanying file Copyright.txt)
// ****************************************************************************
// @HEADER

#include "Tempus_ExplicitTemplateInstantiation.hpp"

#ifdef HAVE_TEMPUS_EXPLICIT_INSTANTIATION
#include "Tempus_StepperOperatorSplit.hpp"
#include "Tempus_StepperOperatorSplit_impl.hpp"

namespace Tempus {

TEMPUS_INSTANTIATE_TEMPLATE_CLASS(StepperOperatorSplit)

// Nonmember constructor
template Teuchos::RCP<StepperOperatorSplit<double> > createStepperOperatorSplit(
    std::vector<Teuchos::RCP<const Thyra::ModelEvaluator<double> > > appModels,
    Teuchos::RCP<Teuchos::ParameterList> pl);

}  // namespace Tempus

#endif
