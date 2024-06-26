#include "triton/Tools/LinearLayout.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/MathExtras.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <iterator>

namespace mlir {
std::ostream &operator<<(std::ostream &os, StringAttr str) {
  os << str.str();
  return os;
}
} // namespace mlir

namespace mlir::triton {
namespace {

using ::llvm::to_vector;
using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::Pair;

using BasesT = LinearLayout::BasesT;

class LinearLayoutTest : public ::testing::Test {
public:
  StringAttr S(StringRef str) { return StringAttr::get(&ctx, str); }

protected:
  MLIRContext ctx;
};

TEST_F(LinearLayoutTest, Empty) {
  LinearLayout layout = LinearLayout::empty();
  EXPECT_THAT(layout.getBases(), IsEmpty());
  EXPECT_THAT(to_vector(layout.getInDimNames()), IsEmpty());
  EXPECT_THAT(to_vector(layout.getOutDimNames()), IsEmpty());
}

TEST_F(LinearLayoutTest, Identity1D) {
  LinearLayout layout =
      LinearLayout::identity1D(32, S("testIns"), S("testOuts"));
  EXPECT_THAT(layout, LinearLayout({{S("testIns"), {{1}, {2}, {4}, {8}, {16}}}},
                                   {S("testOuts")}));
  EXPECT_THAT(to_vector(layout.getInDimNames()), ElementsAre(S("testIns")));
  EXPECT_THAT(to_vector(layout.getOutDimNames()), ElementsAre(S("testOuts")));
  EXPECT_THAT(layout.getInDimSizeLog2(S("testIns")), 5);
  EXPECT_THAT(layout.getOutDimSizeLog2(S("testOuts")), 5);
}

TEST_F(LinearLayoutTest, Identity1DSize1) {
  LinearLayout layout =
      LinearLayout::identity1D(1, S("testIns"), S("testOuts"));
  EXPECT_EQ(layout, LinearLayout({{S("testIns"), {}}}, {S("testOuts")}));
  EXPECT_THAT(to_vector(layout.getInDimNames()), ElementsAre(S("testIns")));
  EXPECT_THAT(to_vector(layout.getOutDimNames()), ElementsAre(S("testOuts")));
  EXPECT_THAT(layout.getInDimSizeLog2(S("testIns")), 0);
  EXPECT_THAT(layout.getOutDimSizeLog2(S("testOuts")), 0);
}

TEST_F(LinearLayoutTest, Zeros1D) {
  LinearLayout layout = LinearLayout::zeros1D(32, S("ins"), S("outs"));
  EXPECT_EQ(layout,
            LinearLayout({{S("ins"), {{0}, {0}, {0}, {0}, {0}}}}, {S("outs")}));
}

TEST_F(LinearLayoutTest, MultiplyIdentity) {
  LinearLayout prod = LinearLayout::identity1D(16, S("in"), S("out")) *
                      LinearLayout::identity1D(32, S("in"), S("out"));
  EXPECT_EQ(prod, LinearLayout(
                      {{S("in"),
                        {{1}, {2}, {4}, {8}, {16}, {32}, {64}, {128}, {256}}}},
                      {S("out")}));
  EXPECT_THAT(to_vector(prod.getInDimNames()), ElementsAre(S("in")));
  EXPECT_THAT(to_vector(prod.getOutDimNames()), ElementsAre(S("out")));
}

TEST_F(LinearLayoutTest, MultiplyDisjoint) {
  LinearLayout prod = LinearLayout::identity1D(32, S("in1"), S("out1")) *
                      LinearLayout::identity1D(16, S("in2"), S("out2"));
  EXPECT_EQ(prod, LinearLayout(
                      {
                          {S("in1"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {16, 0}}},
                          {S("in2"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}}},
                      },
                      {S("out1"), S("out2")}));
  EXPECT_THAT(to_vector(prod.getInDimNames()), ElementsAre(S("in1"), S("in2")));
  EXPECT_THAT(to_vector(prod.getOutDimNames()),
              ElementsAre(S("out1"), S("out2")));
}

TEST_F(LinearLayoutTest, MultiplyByEmpty) {
  LinearLayout prod =
      LinearLayout::empty() * LinearLayout::identity1D(32, S("in"), S("out"));
  EXPECT_EQ(prod, LinearLayout::identity1D(32, S("in"), S("out")));
}

TEST_F(LinearLayoutTest, MultiplyByZeros) {
  LinearLayout prod = LinearLayout::identity1D(8, S("in"), S("out")) *
                      LinearLayout::zeros1D(16, S("in"), S("out"));
  EXPECT_EQ(prod, LinearLayout({{S("in"), {{1}, {2}, {4}, {0}, {0}, {0}, {0}}}},
                               {S("out")}));
}

TEST_F(LinearLayoutTest, MultiplyZerosByDegenerate) {
  LinearLayout prod = LinearLayout::zeros1D(16, S("in"), S("out1")) *
                      LinearLayout({{S("in"), {}}}, {S("out2")});
  EXPECT_EQ(prod, LinearLayout({{S("in"), {{0, 0}, {0, 0}, {0, 0}, {0, 0}}}},
                               {S("out1"), S("out2")}));
}

TEST_F(LinearLayoutTest, MultiplyEmptyIdentityAndZeros) {
  LinearLayout prod = LinearLayout::identity1D(0, S("in"), S("out")) *
                      LinearLayout::zeros1D(4, S("in"), S("out"));
  EXPECT_EQ(prod, LinearLayout({{S("in"), {{0}, {0}}}}, {S("out")}));
}

TEST_F(LinearLayoutTest, MultiplyOverlapping) {
  LinearLayout prod = LinearLayout::identity1D(4, S("in"), S("out1")) *
                      LinearLayout::identity1D(8, S("in"), S("out2"));
  EXPECT_EQ(prod,
            LinearLayout({{S("in"), {{1, 0}, {2, 0}, {0, 1}, {0, 2}, {0, 4}}}},
                         {S("out1"), S("out2")}));
}

TEST_F(LinearLayoutTest, TimesEquals) {
  LinearLayout prod = LinearLayout::empty();
  prod *= LinearLayout::identity1D(32, S("in"), S("out"));
  EXPECT_EQ(prod, LinearLayout::identity1D(32, S("in"), S("out")));
}

TEST_F(LinearLayoutTest, GetOutDimSizeLog2) {
  LinearLayout layout(
      {
          {S("in0"), {{0}, {0}, {0}}},
          {S("in1"), {{1}, {2}}},
      },
      {S("dim0")});
  EXPECT_EQ(layout.getOutDimSizeLog2(S("dim0")), 2);
}

TEST_F(LinearLayoutTest, TransposeOuts) {
  LinearLayout layout = (LinearLayout::identity1D(32, S("in1"), S("out1")) *
                         LinearLayout::identity1D(16, S("in2"), S("out2")))
                            .transposeOuts({S("out2"), S("out1")});
  EXPECT_THAT(to_vector(layout.getOutDimNames()),
              ElementsAre(S("out2"), S("out1")));
  EXPECT_EQ(layout,
            LinearLayout(
                {
                    {S("in1"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {0, 16}}},
                    {S("in2"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}}},
                },
                {S("out2"), S("out1")}));
}

TEST_F(LinearLayoutTest, TransposeOutsDegenerate) {
  LinearLayout layout = (LinearLayout::identity1D(32, S("in1"), S("out1")) *
                         LinearLayout::identity1D(1, S("in2"), S("out2")))
                            .transposeOuts({S("out2"), S("out1")});
  EXPECT_THAT(to_vector(layout.getOutDimNames()),
              ElementsAre(S("out2"), S("out1")));
  EXPECT_EQ(layout,
            LinearLayout(
                {
                    {S("in1"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {0, 16}}},
                    {S("in2"), {}},
                },
                {S("out2"), S("out1")}));
}

TEST_F(LinearLayoutTest, TransposeIns) {
  LinearLayout layout = (LinearLayout::identity1D(32, S("in1"), S("out1")) *
                         LinearLayout::identity1D(16, S("in2"), S("out2")))
                            .transposeIns({S("in2"), S("in1")});
  EXPECT_THAT(to_vector(layout.getInDimNames()),
              ElementsAre(S("in2"), S("in1")));
  EXPECT_EQ(layout,
            LinearLayout(
                {
                    {S("in2"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}}},
                    {S("in1"), {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {16, 0}}},
                },
                {S("out1"), S("out2")}));
}

TEST_F(LinearLayoutTest, EmptyToString) {
  // Mostly I just want to make sure it doesn't crash.
  EXPECT_EQ(LinearLayout::empty().toString(), "(empty layout)\n");
}

TEST_F(LinearLayoutTest, Apply) {
  LinearLayout layout(
      {
          {S("in1"), {{4, 2}, {2, 1}, {1, 0}}},
          {S("in2"), {{1, 2}, {2, 1}}},
      },
      {S("out1"), S("out2")}, /*requireSurjective=*/false);
  EXPECT_THAT(layout.apply({{S("in1"), 0}, {S("in2"), 0}}),
              ElementsAre(Pair(S("out1"), 0), Pair(S("out2"), 0)));
  EXPECT_THAT(layout.apply({{S("in2"), 0}, {S("in1"), 1}}),
              ElementsAre(Pair(S("out1"), 4), Pair(S("out2"), 2)));
  EXPECT_THAT(layout.apply({{S("in2"), 1}, {S("in1"), 0}}),
              ElementsAre(Pair(S("out1"), 1), Pair(S("out2"), 2)));
}

// This is really more of a benchmark than a test.  We're checking that it
// doesn't take so long to run that a human notices and says "hmm".  :)
TEST_F(LinearLayoutTest, ConstructLargeLayout) {
  std::vector<std::vector<int32_t>> pows2;
  for (int i = 0; i < 25; i++) {
    pows2.emplace_back().push_back(1 << i);
  }
  LinearLayout layout({{S("in"), pows2}}, {S("out")});
  (void)layout;
}

TEST_F(LinearLayoutTest, Compose) {
  LinearLayout l1(
      {
          {S("in1"), {{1, 1}, {0, 1}}},
          {S("in2"), {{1, 0}, {1, 2}}},
      },
      {S("out1"), S("out2")});
  LinearLayout l2(
      {
          {S("out1"), {{2, 2}, {1, 0}}},
          {S("out2"), {{1, 1}, {2, 1}}},
      },
      {S("out3"), S("out4")});
  LinearLayout composition = l1.compose(l2);
  EXPECT_EQ(composition,
            LinearLayout(
                {
                    {S("in1"), {{3, 3}, {1, 1}}},
                    {S("in2"), {{2, 2}, {0, 3}}},
                },
                {S("out3"), S("out4")}, /*requireSurjective=*/false));
  EXPECT_FALSE(composition.isSurjective());
}

TEST_F(LinearLayoutTest, Compose4D) {
  LinearLayout l1(
      {{S("in0"), {{1, 0, 0, 0}, {2, 0, 0, 0}}},
       {S("in1"), {{4, 0, 0, 0}, {8, 0, 0, 0}, {16, 0, 0, 0}, {32, 0, 0, 0}}},
       {S("in2"), {{0, 0, 1, 0}, {0, 0, 0, 1}, {0, 0, 0, 2}}},
       {S("in3"), {}}},
      {S("out3"), S("out0"), S("out1"), S("out2")});
  LinearLayout l2(
      {
          {S("out3"),
           {{1, 0, 0, 0},
            {2, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}}},
          {S("out0"), {{0, 1, 0, 0}}},
          {S("out1"), {{0, 0, 1, 0}}},
          {S("out2"), {{0, 0, 0, 1}, {0, 0, 0, 2}}},
      },
      {S("out3"), S("out2"), S("out1"), S("out0")});
  EXPECT_EQ(l1.compose(l2),
            LinearLayout(
                {
                    {S("in0"), {{1, 0, 0, 0}, {2, 0, 0, 0}}},
                    {S("in1"),
                     {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}}},
                    {S("in2"), {{0, 0, 1, 0}, {0, 0, 0, 1}, {0, 0, 0, 2}}},
                    {S("in3"), {}},
                },
                {S("out3"), S("out2"), S("out1"), S("out0")}));
}

TEST_F(LinearLayoutTest, ReshapeIns) {
  LinearLayout ll({{S("in1"), {{1}, {4}, {8}}}, {S("in2"), {{2}}}}, {S("out")});
  EXPECT_EQ(ll.reshapeIns({{S("in3"), {2}}, {S("in4"), {8}}}),
            LinearLayout({{S("in3"), {{1}}}, {S("in4"), {{4}, {8}, {2}}}},
                         {S("out")}));
}

TEST_F(LinearLayoutTest, ReshapeInsDegenerateIn) {
  LinearLayout ll({{S("in1"), {{1}, {4}, {2}}}, {S("in2"), {}}}, {S("out")});
  EXPECT_EQ(
      ll.reshapeIns({{S("in3"), {4}}, {S("in4"), {2}}}),
      LinearLayout({{S("in3"), {{1}, {4}}}, {S("in4"), {{2}}}}, {S("out")}));
}

TEST_F(LinearLayoutTest, ReshapeInsDegenerateOut) {
  LinearLayout ll({{S("in1"), {{1}, {4}}}, {S("in2"), {{2}}}}, {S("out")});
  EXPECT_EQ(
      ll.reshapeIns({{S("in3"), {8}}, {S("in4"), {1}}}),
      LinearLayout({{S("in3"), {{1}, {4}, {2}}}, {S("in4"), {}}}, {S("out")}));
}

TEST_F(LinearLayoutTest, ReshapeInsDegenerateFirstOut) {
  LinearLayout ll({{S("in1"), {{1}, {4}}}, {S("in2"), {{2}}}}, {S("out")});
  EXPECT_EQ(
      ll.reshapeIns({{S("in3"), {1}}, {S("in4"), {8}}}),
      LinearLayout({{S("in3"), {}}, {S("in4"), {{1}, {4}, {2}}}}, {S("out")}));
}

TEST_F(LinearLayoutTest, FlattenIns) {
  LinearLayout ll({{S("in1"), {{1}, {4}, {8}}}, {S("in2"), {{2}}}}, {S("out")});
  EXPECT_EQ(ll.flattenIns(),
            LinearLayout({{S("in1"), {{1}, {4}, {8}, {2}}}}, {S("out")}));
}

TEST_F(LinearLayoutTest, FlattenInsEdgeCases) {
  EXPECT_EQ(LinearLayout({{S("in1"), {}}}, {S("out")}).flattenIns(),
            LinearLayout({{S("in1"), {}}}, {S("out")}));
  EXPECT_EQ(LinearLayout({{S("in1"), {}}}, {}).flattenIns(),
            LinearLayout({{S("in1"), {}}}, {}));
  using BasesArray =
      ArrayRef<std::pair<StringAttr, std::vector<std::vector<int32_t>>>>;
  EXPECT_EQ(LinearLayout(BasesArray{}, {S("out")}).flattenIns(),
            LinearLayout(BasesArray{}, {S("out")}));
  EXPECT_EQ(LinearLayout(BasesArray{}, {}).flattenIns(),
            LinearLayout(BasesArray{}, {}));
}

TEST_F(LinearLayoutTest, ReshapeOuts) {
  LinearLayout ll({{S("in1"), {{1}, {4}, {8}}}, {S("in2"), {{3}}}}, {S("out")});
  EXPECT_EQ(ll.getTotalOutDimSize(), 16);
  EXPECT_EQ(
      ll.reshapeOuts({{S("out2"), {2}}, {S("out3"), {8}}}),
      LinearLayout({{S("in1"), {{1, 0}, {0, 2}, {0, 4}}}, {S("in2"), {{1, 1}}}},
                   {S("out2"), S("out3")}));
}

TEST_F(LinearLayoutTest, ReshapeOutsDegenerateIn) {
  LinearLayout ll({{S("in1"), {{1}, {4}, {2}}}, {S("in2"), {}}}, {S("out")});
  EXPECT_EQ(ll.reshapeOuts({{S("out1"), {4}}, {S("out2"), {2}}}),
            LinearLayout({{S("in1"), {{1, 0}, {0, 1}, {2, 0}}}, {S("in2"), {}}},
                         {S("out1"), S("out2")}));
}

TEST_F(LinearLayoutTest, ReshapeOutsDegenerateOut) {
  LinearLayout ll({{S("in1"), {{1}, {4}}}, {S("in2"), {{2}}}}, {S("out")});
  EXPECT_EQ(ll.reshapeOuts({{S("out1"), {8}}, {S("out2"), {1}}}),
            LinearLayout({{S("in1"), {{1, 0}, {4, 0}}}, {S("in2"), {{2, 0}}}},
                         {S("out1"), S("out2")}));
}

TEST_F(LinearLayoutTest, FlattenOuts) {
  LinearLayout ll({{S("in1"), {{1, 0}, {4, 1}, {8, 4}}}, {S("in2"), {{3, 2}}}},
                  {S("out1"), S("out2")}, /*requireSurjective=*/false);
  EXPECT_EQ(ll.flattenOuts(),
            LinearLayout({{S("in1"), {{1}, {4 + 16}, {8 + 4 * 16}}},
                          {S("in2"), {{3 + 2 * 16}}}},
                         {S("out1")}, /*requireSurjective=*/false));
}

TEST_F(LinearLayoutTest, FlattenOutsEdgeCases) {
  EXPECT_EQ(LinearLayout({{S("in1"), {}}}, {S("out")}).flattenOuts(),
            LinearLayout({{S("in1"), {}}}, {S("out")}));
  EXPECT_EQ(LinearLayout({{S("in1"), {}}}, {}).flattenOuts(),
            LinearLayout({{S("in1"), {}}}, {}));
  using BasesArray =
      ArrayRef<std::pair<StringAttr, std::vector<std::vector<int32_t>>>>;
  EXPECT_EQ(LinearLayout(BasesArray{}, {S("out")}).flattenOuts(),
            LinearLayout(BasesArray{}, {S("out")}));
  EXPECT_EQ(LinearLayout(BasesArray{}, {}).flattenOuts(),
            LinearLayout(BasesArray{}, {}));
}

TEST_F(LinearLayoutTest, InvertAndCompose_Simple) {
  LinearLayout l1({{S("in1"), {{2}, {1}, {4}}}}, {S("out")});
  LinearLayout l2({{S("in2"), {{4}, {1}, {2}}}}, {S("out")});

  // Inverse of l2 is
  //   out(1) => in2=2
  //   out(2) => in2=4
  //   out(4) => in2=1.
  //
  // Composing with l1 gives
  //   l2^-1(l1(1)) = l2^-1(2) = 4
  //   l2^-1(l1(2)) = l2^-1(1) = 2
  //   l2^-1(l1(4)) = l2^-1(4) = 1
  LinearLayout composition = l1.invertAndCompose(l2);
  EXPECT_EQ(composition,
            LinearLayout({{S("in1"), {{4}, {2}, {1}}}}, {S("in2")}));
  // L2 ∘ L2^-1 ∘ L1 == L1.
  EXPECT_EQ(composition.compose(l2), l1);
}

TEST_F(LinearLayoutTest, InvertAndCompose_NonInjective) {
  LinearLayout l1({{S("in1"), {{2}, {1}, {4}}}}, {S("out")});
  LinearLayout l2({{S("in2"), {{0}, {2}, {1}, {4}}}}, {S("out")});

  // The pseudo-inverse of l2 is
  //   out(1) => in2=4
  //   out(2) => in2=2
  //   out(4) => in2=8.
  //
  // Composing with l1 gives
  //   l2^-1(l1(1)) = l2^-1(2) = 2
  //   l2^-1(l1(2)) = l2^-1(0) = 4
  //   l2^-1(l1(4)) = l2^-1(4) = 8
  LinearLayout composition = l1.invertAndCompose(l2);
  EXPECT_EQ(composition, LinearLayout({{S("in1"), {{2}, {4}, {8}}}}, {S("in2")},
                                      /*requireSurjective=*/false));
  EXPECT_FALSE(composition.isSurjective());

  // L2 ∘ L2^-1 ∘ L1 == L1.
  EXPECT_EQ(composition.compose(l2), l1);
}

TEST_F(LinearLayoutTest, InvertAndCompose_SmallerResult) {
  // The domain of l2 is [0,16), but the codomain of the result is only [0,8),
  // because there's no value v in the codomain of l1 such that l2^-1(v) >= 8.
  LinearLayout l1({{S("in1"), {{1}, {2}, {4}}}}, {S("out")});
  LinearLayout l2({{S("in2"), {{4}, {1}, {2}, {8}}}}, {S("out")});
  // Pseudo-inverse of l2 is
  //
  //  out(1) = 2
  //  out(2) = 4
  //  out(4) = 1
  //
  // l1 is the identity, so composing with l1 gives back l2^-1.
  LinearLayout composition = l1.invertAndCompose(l2);
  EXPECT_EQ(composition,
            LinearLayout({{S("in1"), {{2}, {4}, {1}}}}, {S("in2")}));
  EXPECT_EQ(composition.compose(l2), l1);
}

TEST_F(LinearLayoutTest, InvertAndCompose_BroadcastedInDim) {
  LinearLayout l1({{S("in1"), {{2}, {1}, {4}}}, {S("in2"), {{0}}}}, {S("out")});
  LinearLayout l2({{S("in"), {{4}, {1}, {2}}}}, {S("out")});
  // Inverse of l2 is
  //   out(1) = 2
  //   out(2) = 4
  //   out(4) = 1
  //
  // Composing with l1 gives
  //
  //   l2^-1(l1(1, 0)) = l2^-1(2) = 4
  //   l2^-1(l1(2, 0)) = l2^-1(1) = 2
  //   l2^-1(l1(4, 0)) = l2^-1(4) = 1
  //   l2^-1(l1(0, 1)) = l2^-1(0) = 0
  LinearLayout composition = l1.invertAndCompose(l2);
  EXPECT_EQ(composition,
            LinearLayout({{S("in1"), {{4}, {2}, {1}}}, {S("in2"), {{0}}}},
                         {S("in")}));
  EXPECT_EQ(composition.compose(l2), l1);
}

TEST_F(LinearLayoutTest, InvertAndCompose_BroadcastAtBeginningOfSecond) {
  LinearLayout l1({{S("in"), {{1}, {2}, {4}}}}, {S("out")});
  LinearLayout l2({{S("in"), {{0}, {4}, {1}, {2}}}}, {S("out")});
  // Pseudo-inverse of l2 is
  //  out(1) = 4
  //  out(2) = 8
  //  out(4) = 2
  //
  // l1 is the identity, so composing with l1 gives back l2^-1.
  LinearLayout composition = l1.invertAndCompose(l2);
  EXPECT_EQ(composition, LinearLayout({{S("in"), {{4}, {8}, {2}}}}, {S("in")},
                                      /*requireSurjective=*/false));
  EXPECT_EQ(composition.compose(l2), l1);
}

TEST_F(LinearLayoutTest, InvertAndCompose_BroadcastAtEndOfSecond) {
  LinearLayout l1({{S("in1"), {{1}, {2}, {4}}}}, {S("out")});
  LinearLayout l2({{S("in2"), {{4}, {1}, {2}, {0}}}}, {S("out")});
  // Pseudo-inverse of l2 is
  //
  //  out(1) = 2
  //  out(2) = 4
  //  out(4) = 1
  //
  // l1 is the identity, so composing with l1 gives back l2^-1.
  LinearLayout composition = l1.invertAndCompose(l2);
  EXPECT_EQ(composition,
            LinearLayout({{S("in1"), {{2}, {4}, {1}}}}, {S("in2")}));
  EXPECT_EQ(composition.compose(l2), l1);
}

TEST_F(LinearLayoutTest, InvertAndCompose_BroadcastBeginningAndEndOfSecond) {
  LinearLayout l1({{S("in"), {{1}, {2}, {4}}}}, {S("out")});
  LinearLayout l2({{S("in"), {{0}, {4}, {1}, {2}, {0}}}}, {S("out")});
  LinearLayout composition = l1.invertAndCompose(l2);
  EXPECT_EQ(composition, LinearLayout({{S("in"), {{4}, {8}, {2}}}}, {S("in")},
                                      /*requireSurjective=*/false));
  EXPECT_EQ(composition.compose(l2), l1);
}

TEST_F(LinearLayoutTest, InvertAndCompose_Multidim) {
  LinearLayout l1(
      {{S("in1"), {{1, 0}, {0, 1}, {2, 0}, {3, 2}}}, {S("in2"), {{2, 2}}}},
      {S("out1"), S("out2")});
  LinearLayout l2({{S("in3"), {{0, 1}, {1, 0}, {0, 0}, {0, 2}, {2, 1}}}},
                  {S("out2"), S("out1")});

  LinearLayout c1 = l1.invertAndCompose(l2);
  EXPECT_EQ(c1.compose(l2), l1.transposeOuts(l2.getOutDimNames()));

  LinearLayout c2 = l2.invertAndCompose(l1);
  EXPECT_EQ(c2.compose(l1), l2.transposeOuts(l1.getOutDimNames()));
}

TEST_F(LinearLayoutTest, NumConsecutiveInOut) {
  EXPECT_EQ(
      1,
      LinearLayout::identity1D(1, S("in"), S("out")).getNumConsecutiveInOut());
  EXPECT_EQ(
      4,
      LinearLayout::identity1D(4, S("in"), S("out")).getNumConsecutiveInOut());
  EXPECT_EQ(4, (LinearLayout::identity1D(4, S("in1"), S("out")) *
                LinearLayout::identity1D(8, S("in2"), S("out")))
                   .getNumConsecutiveInOut());
  EXPECT_EQ(4, (LinearLayout::identity1D(4, S("in"), S("out1")) *
                LinearLayout::identity1D(8, S("in"), S("out2")))
                   .getNumConsecutiveInOut());
  EXPECT_EQ(1, (LinearLayout::zeros1D(4, S("in"), S("out1")) *
                LinearLayout::identity1D(4, S("in"), S("out2")))
                   .getNumConsecutiveInOut());
  EXPECT_EQ(1, LinearLayout({{S("in"), {{1}, {2}, {4}, {9}}}}, {S("out")})
                   .getNumConsecutiveInOut());
  EXPECT_EQ(2, LinearLayout({{S("in"), {{1}, {2}, {4}, {10}}}}, {S("out")})
                   .getNumConsecutiveInOut());
  EXPECT_EQ(2, LinearLayout({{S("in"), {{1}, {4}, {2}}}}, {S("out")})
                   .getNumConsecutiveInOut());
  EXPECT_EQ(2, LinearLayout(
                   {
                       {S("in"), {{1}, {2}, {4}}},
                       {S("in2"), {{8}, {18}}},
                   },
                   {S("out")})
                   .getNumConsecutiveInOut());
}

} // anonymous namespace
} // namespace mlir::triton
