# SpecNaacl: FastGRPO và ReflexOPD

Core dùng trực tiếp FastGRPO gốc trong `sources/FastGRPO`, được sao chép từ
`../FastGRPO-main` và kiểm tra SHA256. Chỉ có `METHOD=fastgrpo` và
`METHOD=opd_reflex`.

Cả hai dùng FastGRPO DraftModel (EagleFS, một DraftDecoderLayer,
nhánh dự đoán feature/logits), embedding và lm_head của target, full target
vocabulary, cùng checkpoint pretrain và objective online. Baseline giữ nguyên
hot path upstream. ReflexOPD thêm correction rank 8, Top16 và cập nhật GPU;
KV/workspace tối ưu chỉ nằm trong nhánh OPD.

Pretrain mặc định ShareGPT, 5 epochs. Training mặc định SimpleLR
`simplelr_abel_level3to5`, target LR `1e-6`, draft LR `1e-4`, draft accumulation `1`.
Không có runtime dependency vào SpecForge/EAGLE3/SGLang hay vocabulary mapping.
Checkpoint SpecForge cũ cần được thay bằng checkpoint pretrain FastGRPO mới.

- [Cách cài môi trường](ENVIRONMENT.md)
- [Lệnh pretrain/train/resume/benchmark](huongdanchay.md)
- [Kiến trúc, đối chiếu source và kết quả kiểm thử](FASTGRPO_REWRITE.md)
- [Chi tiết ReflexOPD](METHOD_OPD_REFLEX.md)
- [Autotuning proposal](OPD_AUTOTUNING.md)

Các bản runtime, test và tài liệu trước rewrite được giữ riêng trong
`legacy_tests/specforge/`; chúng không thuộc bộ test hay runtime hiện tại.
