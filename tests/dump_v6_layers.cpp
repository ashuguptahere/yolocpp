// Dev-only: dump per-stage activation stats + raw P2..P6 backbone outputs
// for v6l6 parity debugging vs upstream Python. Not a ctest.

#include "yolocpp/inference/letterbox.hpp"
#include "yolocpp/models/yolo6.hpp"
#include "yolocpp/serialization/pt_loader.hpp"
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>

using namespace yolocpp;

static void dump(const std::string& name, const torch::Tensor& t) {
  auto cpu = t.detach().cpu().to(torch::kFloat32);
  std::cerr << name << " " << cpu.sizes() << " mean=" << cpu.mean().item<float>()
            << " std=" << cpu.std().item<float>()
            << " min=" << cpu.min().item<float>()
            << " max=" << cpu.max().item<float>() << "\n";
}

int main(int argc, char** argv) {
  if (argc < 2) { std::cerr << "usage: dump_v6_layers weights.pt\n"; return 1; }
  models::Yolo6Scale scale = models::kYolo6l;
  models::Yolo6 m(80, scale, 16, /*p6=*/true);
  auto sd = serialization::load_state_dict(argv[1]);
  m->load_from_state_dict(sd.entries);
  m->eval();

  cv::Mat img = cv::imread("/home/ashu/Desktop/yolocpp/data/bus.jpg");
  auto lb = inference::letterbox(img, 1280);
  auto x = inference::image_to_tensor(lb.img).unsqueeze(0).contiguous();

  torch::NoGradGuard ng;
  auto* bp = m->backbone_p6.get();
  auto stem_fwd = [&](const torch::Tensor& t, models::RepConv& rp, models::ConvBNReLU& cbr) {
    return rp ? rp->forward(t) : cbr->forward(t);
  };
  auto block_fwd = [&](const torch::Tensor& t, models::RepBlock& rb, models::BepC3& bep) {
    return rb ? rb->forward(t) : bep->forward(t);
  };
  auto y = stem_fwd(x, bp->stem_rep, bp->stem_cbr);                                      dump("stem", y);
  y = stem_fwd(y, bp->ERBlock_2_down_rep, bp->ERBlock_2_down_cbr);                       dump("ERBlock_2_down", y);
  auto p2 = block_fwd(y, bp->ERBlock_2_block_rb, bp->ERBlock_2_block_bep);               dump("ERBlock_2_block (P2)", p2);
  auto p3_in = stem_fwd(p2, bp->ERBlock_3_down_rep, bp->ERBlock_3_down_cbr);
  auto p3 = block_fwd(p3_in, bp->ERBlock_3_block_rb, bp->ERBlock_3_block_bep);           dump("ERBlock_3_block (P3)", p3);
  auto p4_in = stem_fwd(p3, bp->ERBlock_4_down_rep, bp->ERBlock_4_down_cbr);
  auto p4 = block_fwd(p4_in, bp->ERBlock_4_block_rb, bp->ERBlock_4_block_bep);           dump("ERBlock_4_block (P4)", p4);
  auto p5_in = stem_fwd(p4, bp->ERBlock_5_down_rep, bp->ERBlock_5_down_cbr);
  auto p5 = block_fwd(p5_in, bp->ERBlock_5_block_rb, bp->ERBlock_5_block_bep);           dump("ERBlock_5_block (P5)", p5);
  auto p6_in = stem_fwd(p5, bp->ERBlock_6_down_rep, bp->ERBlock_6_down_cbr);
  auto p6_block = block_fwd(p6_in, bp->ERBlock_6_block_rb, bp->ERBlock_6_block_bep);     dump("ERBlock_6_block", p6_block);
  auto p6_out = bp->ERBlock_6_cspsppf ? bp->ERBlock_6_cspsppf->forward(p6_block)
                                       : bp->ERBlock_6_simsppf->forward(p6_block);      dump("ERBlock_6_sppf (P6)", p6_out);
  for (int i = 0; i < 5; ++i) {
    torch::Tensor t = (i==0)?p2:(i==1)?p3:(i==2)?p4:(i==3)?p5:p6_out;
    auto cpu = t.cpu().to(torch::kFloat32).contiguous();
    std::ofstream f("/tmp/yolocpp_v6l6/cpp_p" + std::to_string(i+2) + ".bin", std::ios::binary);
    f.write((const char*)cpu.data_ptr(), cpu.numel() * sizeof(float));
  }

  // Now run the full neck + head and dump per-level head outputs.
  auto* np = m->neck_p6.get();
  auto fpn0      = np->reduce_layer0(p6_out);                                          dump("reduce_layer0", fpn0);
  auto fused5    = np->Bifusion0(p5, p4, fpn0);                                        dump("Bifusion0", fused5);
  auto out_p5_td = block_fwd(fused5, np->Rep_p5_rb, np->Rep_p5_bep);                   dump("Rep_p5", out_p5_td);
  auto fpn1      = np->reduce_layer1(out_p5_td);                                       dump("reduce_layer1", fpn1);
  auto fused4    = np->Bifusion1(p4, p3, fpn1);                                        dump("Bifusion1", fused4);
  auto out_p4_td = block_fwd(fused4, np->Rep_p4_rb, np->Rep_p4_bep);                   dump("Rep_p4", out_p4_td);
  auto fpn2      = np->reduce_layer2(out_p4_td);                                       dump("reduce_layer2", fpn2);
  auto fused3    = np->Bifusion2(p3, p2, fpn2);                                        dump("Bifusion2", fused3);
  auto out_p3    = block_fwd(fused3, np->Rep_p3_rb, np->Rep_p3_bep);                   dump("Rep_p3 (P3 neck)", out_p3);
  auto d2     = np->downsample2(out_p3);
  auto out_p4 = block_fwd(torch::cat({d2, fpn2}, 1), np->Rep_n4_rb, np->Rep_n4_bep);   dump("Rep_n4 (P4 neck)", out_p4);
  auto d1     = np->downsample1(out_p4);
  auto out_p5n = block_fwd(torch::cat({d1, fpn1}, 1), np->Rep_n5_rb, np->Rep_n5_bep);  dump("Rep_n5 (P5 neck)", out_p5n);
  auto d0     = np->downsample0(out_p5n);
  auto out_p6 = block_fwd(torch::cat({d0, fpn0}, 1), np->Rep_n6_rb, np->Rep_n6_bep);   dump("Rep_n6 (P6 neck)", out_p6);

  std::vector<torch::Tensor> neck_outs = {out_p3, out_p4, out_p5n, out_p6};
  for (int i = 0; i < 4; ++i) {
    auto cpu = neck_outs[i].cpu().to(torch::kFloat32).contiguous();
    std::ofstream f("/tmp/yolocpp_v6l6/cpp_neck" + std::to_string(i) + ".bin", std::ios::binary);
    f.write((const char*)cpu.data_ptr(), cpu.numel() * sizeof(float));
  }
  return 0;
}
