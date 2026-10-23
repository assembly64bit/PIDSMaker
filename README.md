# PIDSMaker baselines: THEIA_E3 / TRACE_E3 / THEIA_E5
Link code để chạy có setup và kèm dataset : https://www.kaggle.com/datasets/nguynnhtmlp11a13/pidsmaker.

Chạy 4 thí nghiệm baseline (Velox, Orthrus) trên PIDSMaker, mỗi lần chạy đo CPU / RAM / VRAM, lưu log và metric (precision, recall, F-score...), tự dọn đĩa và gửi thông báo.

| Tên | Model | Kiểu | Train | Test | DB cần |
|---|---|---|---|---|---|
| `A_velox` | Velox | same-host | `THEIA_E3_CUT` | `THEIA_E3_CUT` | `theia_e3_cut` |
| `A_orthrus` | Orthrus | same-host | `THEIA_E3_CUT` | `THEIA_E3_CUT` | `theia_e3_cut` |
| `B_velox` | Velox | cross-host | `THEIA_E3` + `TRACE_E3` | `THEIA_E5` | `theia_e3`, `trace_e3`, `theia_e5` |
| `B_orthrus` | Orthrus | cross-host | `THEIA_E3` + `TRACE_E3` | `THEIA_E5` | như trên |

Bài B dùng tham số tuned của `THEIA_E3` (host dùng để train). Cờ cross-host nằm trong biến `CROSS` của `run_all.sh`:
`multi_dataset=THEIA_E3,TRACE_E3`, `multi_dataset_training=True` (featurization và batching), `multi_dataset_primary_test_only=True`. Dataset gõ ở dòng lệnh (`THEIA_E5`) luôn là dataset test, train/val của nó không được dùng để học.

## 1. Yêu cầu

- Linux hoặc WSL2, GPU NVIDIA đời Ampere/Hopper trở lên, driver hỗ trợ CUDA >= 12.1 (`nvidia-smi`). GPU Blackwell chưa chạy được với torch 2.1.2. Không có GPU thì thêm `COMMON="--cpu"`.
- PostgreSQL 17, chạy ở `127.0.0.1:5432`, user và password `postgres`.
- Python 3.9 (conda), torch 2.1.2 + CUDA 12.1.
- Đĩa trống: 4 database khoảng 95 GB (`theia_e3` 12 GB, `theia_e3_cut` 12 GB, `trace_e3` 34 GB bản đã cắt, `theia_e5` 36 GB), cộng artifacts. Nên có từ 200 GB.
- Nếu dùng WSL: giới hạn thật là ổ C: của Windows. `df` trong WSL chỉ thấy đĩa ảo nên không đáng tin.

## 2. Cài đặt

```bash
# Hệ thống + Postgres 17
sudo apt-get update -y
sudo apt-get install -y --no-install-recommends git curl wget ca-certificates gnupg lsb-release graphviz postgresql-common tmux pigz unzip rsync
sudo /usr/share/postgresql-common/pgdg/apt.postgresql.org.sh -y
sudo apt-get install -y postgresql-17
sudo -u postgres psql -c "ALTER USER postgres PASSWORD 'postgres';"
pg_isready -h 127.0.0.1

# conda + env
wget -q -O /tmp/miniforge.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash /tmp/miniforge.sh -b -p $HOME/miniforge3
source $HOME/miniforge3/etc/profile.d/conda.sh
conda create -y -n pids -c conda-forge --override-channels python=3.9
conda activate pids

# thư viện Python
pip install --upgrade pip
pip install psycopg2-binary tqdm scikit-learn==1.6.1 networkx==3.2.1 xxhash==3.2.0 graphviz==0.20.1 psutil scipy==1.13.1 matplotlib==3.9.4 wandb==0.24.2 chardet==5.2.0 nltk==3.8.1 igraph==0.11.5 cairocffi==1.7.0 wget==3.2
pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 torchaudio==2.1.2+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
pip install torch_geometric==2.6.1 --no-cache-dir
pip install pyg_lib==0.4.0 torch_scatter==2.1.2 torch_sparse==0.6.18 torch_cluster==1.6.3 torch_spline_conv==1.2.2 -f https://data.pyg.org/whl/torch-2.1.0+cu121.html --no-cache-dir
pip install gensim==4.4.0 pytz==2024.1 pandas==2.3.3 yacs==0.1.8 transformers==4.39.3
pip install numpy==1.26.4
pip install gdown==5.2.0 umap-learn==0.5.6 flask==3.0.3 pytest==8.3.5
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

# kiểm tra GPU
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # phải ra True
pip install -e .
```

## 3. Chuẩn bị dữ liệu

Dump gốc lấy từ thư mục Google Drive của PIDSMaker (xem docs của PIDSMaker). Bản đã xử lý của dự án này nằm trong Kaggle dataset `nguynnhtmlp11a13/pidsmaker`:

```bash
pip install kaggle
export KAGGLE_USERNAME="nguynnhtmlp11a13"
export KAGGLE_KEY="KGAT_9a2c1042e6c49519f5e8ed1ea10d6dfe"      # kaggle.com/settings -> API
kaggle datasets download -d nguynnhtmlp11a13/pidsmaker -p ~/kdata --unzip
```

Nạp từng DB (đừng Ctrl+C giữa chừng, nạp lỗi thì `dropdb` rồi làm lại):

```bash
export PGPASSWORD=postgres; P="-U postgres -h 127.0.0.1"
for db in theia_e3 trace_e3 theia_e3_cut; do
  createdb $P $db
  pg_restore $P -d $db --no-owner -j 4 data/$db.dump
done
```

`theia_e5` không cần nạp tay: nếu có `data/theia_e5.dump` thì `run_all.sh` tự nạp, chưa có thì tự tải (xem mục 4).

### Xử lý các DB (đã làm sẵn trong bản dump trên Kaggle)

| DB | Xử lý |
|---|---|
| `theia_e3_cut` | Tạo bằng `prepare_cutoff_db.py` (cắt theo cutoff, xóa node attack ở phần train/val). Chạy `python prepare_cutoff_db.py --help`. |
| `theia_e3` | Xóa node attack (theo file GT) khỏi mọi ngày bằng `scrub_inplace.py`. Vì mất cả node attack ở ngày test nên **không dùng để chấm điểm**, chỉ để train. |
| `trace_e3` | Chỉ giữ các ngày trước 2018-04-10 (attack đầu tiên nhắm vào TRACE là 04-10), rồi xóa node attack. Ngày test của TRACE_E3 trong DB này **trống**, bài B không dùng chúng. |

Chạy lại bước xóa node attack (mặc định chỉ đếm, thêm `--apply` mới xóa thật):

```bash
python scrub_inplace.py --dataset THEIA_E3 --db theia_e3
python scrub_inplace.py --dataset TRACE_E3 --db trace_e3
python scrub_inplace.py --dataset THEIA_E3 --db theia_e3 --apply
python scrub_inplace.py --dataset TRACE_E3 --db trace_e3 --apply
```

Script xóa các node thuộc file GT (trong `subject_node_table`, `file_node_table`, `netflow_node_table`) và mọi event nối vào chúng, trong một transaction rồi tự kiểm tra lại.

## 4. Chạy

```bash
chmod +x run_all.sh
./run_all.sh                          # cả 4 bài
ONLY="A_velox" ./run_all.sh           # một bài (nên chạy thử A_velox trước)
ONLY="B_velox B_orthrus" ./run_all.sh
```

Chạy dài nên dùng `tmux` (`tmux new -s run`, `Ctrl+B` rồi `D` để thoát, `tmux attach -t run` để quay lại).

Trước khi chạy, script tự kiểm tra DB: bài A cần `theia_e3_cut`, bài B cần `theia_e3`, `trace_e3` và tự lo `theia_e5`. Thiếu DB nào ngoài `theia_e5` thì dừng ngay với thông báo rõ.

Với `theia_e5`: nếu có sẵn `data/theia_e5.dump` thì chỉ nạp, nếu không thì cần `GDRIVE_TOKEN` (access token Google, scope `drive.readonly`, lấy ở https://developers.google.com/oauthplayground/) và `download_datasets.sh` trong repo để tải (5.8 GB nén, khoảng 36 GB sau khi nạp):

```bash
GDRIVE_TOKEN=ya29.xxxx ONLY="B_velox" ./run_all.sh
```

### Biến môi trường

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `ONLY` | (rỗng = cả 4) | Danh sách bài cần chạy |
| `OUT` | `results` | Thư mục log và metric |
| `ARTIFACT_DIR` | `$HOME/artifacts` | Kết quả trung gian |
| `DUMP_DIR` | `data` | Nơi tìm `theia_e5.dump` |
| `DB_HOST`, `DB_PORT` | `127.0.0.1`, `5432` | Postgres |
| `PGUSER`, `PGPASSWORD` | `postgres` | Tài khoản Postgres |
| `COMMON` | (rỗng) | Tham số thêm cho mọi lần chạy, ví dụ `--cpu` |
| `VELOX_ARGS`, `ORTHRUS_ARGS` | tuned THEIA_E3 | Tham số huấn luyện |
| `CLEAN` | `after` | `after` = dọn artifacts sau mỗi bài thành công, `none` = giữ hết |
| `KEEP_RE` | `^(evaluation\|triage\|wandb)$` | Thư mục artifacts được giữ khi dọn |
| `MIN_FREE_GB` | `25` | Đĩa trống tối thiểu trước mỗi bài, thấp hơn thì dọn, vẫn thiếu thì dừng |
| `GDRIVE_TOKEN` | (rỗng) | Token tải `theia_e5` |
| `NOTIFY_TELEGRAM_TOKEN`, `NOTIFY_TELEGRAM_CHAT` | (rỗng) | Thông báo Telegram |
| `NOTIFY_WEBHOOK` | (rỗng) | Webhook Slack hoặc Discord |
| `NOTIFY_TOAST` | `1` | Toast Windows (chỉ WSL), đặt `0` để tắt |

`run_all.sh` tự `cd` vào thư mục chứa nó nên đặt repo ở đâu cũng chạy được.

## 5. Kết quả

Trong `results/` (hoặc `$OUT`):

- `run_all_log.txt`: toàn bộ output của cả phiên chạy.
- `<tên bài>.log`: log đầy đủ từng bài (stdout và stderr).
- `<tên bài>_samples.csv`: CPU, RAM, GPU mỗi giây. `<tên bài>_summary.txt`: tổng kết tài nguyên.
- `metrics_log.txt`: sau mỗi bài, bảng tp/fp/fn/precision/recall/fscore và thời gian chạy.
- `results_summary.csv`: bảng metric gộp.
- `notifications.txt`: mọi thông báo đã gửi.

Dọn đĩa chỉ chạy sau bài thành công và giữ `evaluation`, `triage`, `wandb`, nên metric vẫn còn. Bài lỗi thì giữ artifacts để debug. Đổi lại, bài sau không dùng lại được cache của bài trước. Muốn giữ cache thì `CLEAN=none`.

## 6. Kiểm tra

```bash
bash verify_final.sh 2>&1 | tee verify_out.txt
```

Kết quả đúng:

- `theia_e3` còn 29,203,665 event, `trace_e3` còn 25,244,673 event.
- `scrub_inplace.py` chạy không `--apply` báo 0 event cần xóa.
- `check_train_contamination.py` ra `SACH` cho `THEIA_E3` và `TRACE_E3`. Với `THEIA_E3_CUT` ra `SACH` ở train/val và vẫn còn event attack ở ngày test.

Khi bài B chạy, kiểm tra log:

```bash
grep -n "WARNING.*primary_test_only" results/B_velox.log   # không được ra dòng nào
grep -n "Multi-dataset order" results/B_velox.log          # xem thứ tự dataset
python tests/test_cross_host_multi_dataset.py              # test độc lập, chỉ cần torch
```

Nếu có cảnh báo `primary_test_only` thì train/val của `THEIA_E5` bị kéo vào huấn luyện, thí nghiệm không còn là cross-host thuần.

## 7. Lỗi thường gặp

| Lỗi | Cách xử lý |
|---|---|
| `pg_restore: error: ... already exists` | DB nạp dở trước đó. `dropdb --force <db>` rồi nạp lại. |
| `SSL error: unexpected eof` hoặc `Input/output error` lúc nạp | Hết đĩa. Giải phóng đĩa (với WSL là ổ C:), `dropdb --force` DB dở rồi nạp lại. |
| `database ... is being accessed by other users` | Có kết nối treo. `SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname='<db>';` rồi `dropdb --force`. |
| `pg_restore` từ chối dump | Dump tạo bằng Postgres mới hơn bản đang cài. Dùng Postgres 17 hoặc dump lại bằng đúng phiên bản. |
| Bài B báo lỗi ở bước dựng graph của `TRACE_E3` | Ngày test của `TRACE_E3` trống. Bỏ các ngày đó khỏi config dataset `TRACE_E3`. |
| `torch.cuda.is_available()` ra `False` | Driver chưa đúng hoặc GPU không được hỗ trợ. Kiểm tra `nvidia-smi` và phiên bản CUDA. |
| `collect_results loi` trong `metrics_log.txt` | Chạy trong đúng môi trường conda `pids` (cần torch). |

## 8. Cấu trúc

```
run_all.sh                     chạy 4 bài, kiểm tra DB, log, thông báo, dọn đĩa
resource_monitor.py            đo CPU / RAM / GPU quanh mỗi lệnh
collect_results.py             gom metric từ artifacts
prepare_cutoff_db.py           tạo DB cắt theo cutoff (bài A)
scrub_inplace.py               xóa node attack và event nối vào chúng, tại chỗ
check_train_contamination.py   kiểm tra node attack có nằm trong ngày train/val không
make_test_gt.py                tạo ground truth cho phần test
verify_final.sh                kiểm tra lần cuối các DB
vm_setup.sh                    cài đặt môi trường
tests/                         test cross-host
pidsmaker/, config/            mã PIDSMaker và cấu hình
final_checksums.txt            sha256 của các file trong gói
```

## 9. Giới hạn đã biết

- Kết quả kiểm tra nhiễm chỉ dựa vào các attack đã được gán nhãn trong file GT. Hoạt động độc hại chưa được gán nhãn thì không phát hiện được.
- Xóa node GT cũng xóa mọi event benign nối vào node đó (ví dụ file hệ thống dùng chung, hoặc netflow placeholder ở THEIA), nên baseline mất một phần hành vi bình thường.
- `trace_e3` chỉ dùng để train. Không đánh giá attack trên `trace_e3` và `theia_e3` đã xóa node.
