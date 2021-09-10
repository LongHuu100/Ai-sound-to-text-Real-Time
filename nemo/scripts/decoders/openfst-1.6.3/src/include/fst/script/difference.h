// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#ifndef FST_SCRIPT_DIFFERENCE_H_
#define FST_SCRIPT_DIFFERENCE_H_

#include <fst/difference.h>
#include <fst/script/arg-packs.h>
#include <fst/script/compose.h>  // for ComposeFilter
#include <fst/script/fst-class.h>

namespace fst {
namespace script {

using DifferenceArgs1 = args::Package<const FstClass &, const FstClass &,
                                      MutableFstClass *, ComposeFilter>;

template <class Arc>
void Difference(DifferenceArgs1 *args) {
  const Fst<Arc> &ifst1 = *(args->arg1.GetFst<Arc>());
  const Fst<Arc> &ifst2 = *(args->arg2.GetFst<Arc>());
  MutableFst<Arc> *ofst = args->arg3->GetMutableFst<Arc>();
  Difference(ifst1, ifst2, ofst, ComposeOptions(args->arg4));
}

using DifferenceArgs2 =
    args::Package<const FstClass &, const FstClass &, MutableFstClass *,
                  const ComposeOptions &>;

template <class Arc>
void Difference(DifferenceArgs2 *args) {
  const Fst<Arc> &ifst1 = *(args->arg1.GetFst<Arc>());
  const Fst<Arc> &ifst2 = *(args->arg2.GetFst<Arc>());
  MutableFst<Arc> *ofst = args->arg3->GetMutableFst<Arc>();
  Difference(ifst1, ifst2, ofst, ComposeOptions(args->arg4));
}

void Difference(const FstClass &ifst1, const FstClass &ifst2,
                MutableFstClass *ofst, ComposeFilter compose_filter);

void Difference(const FstClass &ifst1, const FstClass &ifst2,
                MutableFstClass *ofst,
                const ComposeOptions &opts = fst::script::ComposeOptions());

}  // namespace script
}  // namespace fst

#endif  // FST_SCRIPT_DIFFERENCE_H_
