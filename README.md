# PIDSMaker baselines: THEIA_E3 / TRACE_E3 / THEIA_E5

Link code để chạy, có setup và kèm dataset: <https://www.kaggle.com/datasets/nguynnhtmlp11a13/pidsmaker>

> **Chạy nhanh:** chỉ cần dán **một ô bash duy nhất** ở mục 10 ([#1] STEP 1 - SETUP + CHECK GPU + ENV + RUN PIPELINE). Ô đó tự kiểm tra hệ thống/GPU, cài đặt môi trường, Postgres/conda/torch, ghi toàn bộ script còn lại, rồi nạp DB, chạy 4 bài, gom metric, dọn đĩa, verify và gửi thông báo.

> File này gồm 2 phần:
> 1. Hướng dẫn đầy đủ để cài đặt, nạp dữ liệu, chạy 4 thí nghiệm và kiểm tra kết quả.
> 2. **Một ô bash duy nhất ở mục 10**: chứa toàn bộ script điều phối, đo tài nguyên, chuẩn bị DB, test, ground truth và patch lõi PIDSMaker cần thiết để tái lập pipeline.

Chạy 4 thí nghiệm baseline (Velox, Orthrus) trên PIDSMaker, mỗi lần chạy đo CPU / RAM / VRAM, lưu log và metric (precision, recall, F-score...), tự dọn đĩa và gửi thông báo.

| Tên | Model | Kiểu | Train | Test | DB cần |
|---|---|---|---|---|---|
| `A_velox` | Velox | same-host | `THEIA_E3_CUT` | `THEIA_E3_CUT` | `theia_e3_cut` |
| `A_orthrus` | Orthrus | same-host | `THEIA_E3_CUT` | `THEIA_E3_CUT` | `theia_e3_cut` |
| `B_velox` | Velox | cross-host | `THEIA_E3` + `TRACE_E3` | `THEIA_E5` | `theia_e3`, `trace_e3`, `theia_e5` |
| `B_orthrus` | Orthrus | cross-host | `THEIA_E3` + `TRACE_E3` | `THEIA_E5` | như trên |

Bài B train trên `THEIA_E3` + `TRACE_E3`, test trên `THEIA_E5`. Siêu tham số (dropout, learning rate, hidden dim, số epoch...) dùng bộ đã tune sẵn cho `THEIA_E3` trong docs của PIDSMaker (`docs/docs/tuned_systems.md`), không dùng bộ tune của `THEIA_E5` vì bộ đó được chỉnh ngay trên host test, dùng sẽ rò rỉ thông tin test. Cờ cross-host nằm trong biến `CROSS` của `run_all.sh`:
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

Toàn bộ bước trên nằm trong `vm_setup.sh`; ô bash ở mục 10 cũng tự chạy file này đầu tiên:

```bash
chmod +x vm_setup.sh && ./vm_setup.sh
```

## 3. Chuẩn bị dữ liệu

Dump gốc lấy từ thư mục Google Drive của PIDSMaker (xem docs của PIDSMaker). Bản đã xử lý của dự án này nằm trong Kaggle dataset `nguynnhtmlp11a13/pidsmaker`:

```bash
pip install kaggle
# Token Kaggle (KGAT_...): đặt qua biến môi trường, KHÔNG ghi vào file/commit.
# Ví dụ: export KAGGLE_API_TOKEN="$(cat ~/.kaggle/access_token)"
export KAGGLE_API_TOKEN="<token của bạn>"
export KAGGLE_USERNAME="${KAGGLE_USERNAME:-nguynnhtmlp11a13}"
kaggle datasets download -d nguynnhtmlp11a13/pidsmaker -p ~/kdata --unzip
```

> **Cảnh báo bảo mật:** token là mật khẩu. Không dán vào file `.md`, không commit. Nếu token đã bị lộ (dán lên chat, commit...), vào <https://www.kaggle.com/settings> revoke và tạo token mới. Cách an toàn: lưu vào `~/.kaggle/access_token` với `chmod 600`, rồi `export KAGGLE_API_TOKEN="$(cat ~/.kaggle/access_token)"`. Ô bash ở mục 10 tự đọc file này nếu có.

`run_all.sh` tự nạp DB còn thiếu từ `data/*.dump` (mặc định `RESTORE=missing`). Muốn nạp tay:

```bash
export PGPASSWORD=postgres; P="-U postgres -h 127.0.0.1"
for db in theia_e3 trace_e3 theia_e3_cut; do
  createdb $P $db
  pg_restore $P -d $db --no-owner -j 4 data/$db.dump
done
```

Không nên Ctrl+C giữa chừng khi `pg_restore`. Nếu nạp lỗi: `dropdb --force <db>` rồi làm lại. `theia_e5` cũng được `run_all.sh` tự nạp khi có `data/theia_e5.dump`, chưa có thì tự tải (xem mục 4).

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
SEEDS="1" ./run_all.sh                # thêm 4 bản seed -> lấp đầy 8 GPU
PARALLEL=0 ./run_all.sh               # chạy tuần tự
DRY_RUN=1 ./run_all.sh                # chỉ in lệnh, không chạy
RESTORE=force ./run_all.sh            # drop + nạp lại DB từ data/*.dump
```

`run_all.sh` làm toàn bộ pipeline trong một lần:

1. Kiểm tra môi trường (python/torch, Postgres).
2. Nạp DB còn thiếu từ `data/*.dump` (`RESTORE=missing`) hoặc nạp lại (`RESTORE=force`).
3. Kiểm tra/tạo index `timestamp_rec` cho các DB cần dùng.
4. Chạy các bài: tự song song nếu máy có >= 2 GPU (`PARALLEL=auto`), mỗi bài một GPU qua `CUDA_VISIBLE_DEVICES`.
5. Gom metric, dọn artifacts, chạy `verify_final.sh` và gửi thông báo.

Thiếu dump của DB nào (trừ `theia_e5` có thể tải) thì script dừng ngay với thông báo rõ.

Chạy dài nên dùng `tmux` (`tmux new -s run`, `Ctrl+B` rồi `D` để thoát, `tmux attach -t run` để quay lại).

`run_parallel.sh` là wrapper tương thích ngược: `bash run_parallel.sh` tương đương `PARALLEL=1 OUT=results_parallel bash run_all.sh`.

Với `theia_e5`: nếu có sẵn `data/theia_e5.dump` thì chỉ nạp, nếu không thì cần `GDRIVE_TOKEN` (access token Google, scope `drive.readonly`, lấy ở <https://developers.google.com/oauthplayground/>) và `download_datasets.sh` trong repo để tải (5.8 GB nén, khoảng 36 GB sau khi nạp):

```bash
GDRIVE_TOKEN=ya29.xxxx ONLY="B_velox" ./run_all.sh
```

### Biến môi trường

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `ONLY` | (rỗng = cả 4) | Danh sách bài cần chạy |
| `OUT` | `results` | Thư mục log và metric |
| `ARTIFACT_DIR` | `$HOME/artifacts` | Kết quả trung gian |
| `DUMP_DIR` | `data` | Nơi tìm `<db>.dump` |
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
| `PARALLEL` | `auto` | `auto` = song song nếu >= 2 GPU; `1` = luôn song song; `0` = tuần tự |
| `NUM_GPUS` | tự dò | Số GPU vật lý dùng để chia bài |
| `SEEDS` | (rỗng) | Ví dụ `"1"` hoặc `"1 2"`: thêm bản seed cho mỗi bài |
| `ISOLATE` | `0` | `1` = ép mỗi bài một hash riêng (`--restart_from_scratch`) |
| `INTERVAL` | `1` | Giây giữa 2 lần lấy mẫu tài nguyên |
| `RESTORE` | `missing` | `missing` = chỉ nạp DB thiếu; `force` = drop + nạp lại |
| `PG_RESTORE_JOBS` | `4` | Số job song song của `pg_restore` |
| `SKIP_INDEX` | `0` | `1` = bỏ qua kiểm tra/tạo index `timestamp_rec` |
| `VERIFY` | `1` | `1` = chạy `verify_final.sh` cuối cùng |
| `DRY_RUN` | `0` | `1` = chỉ in lệnh |
| `SKIP_SETUP` | `0` | `1` = bỏ qua `vm_setup.sh` (máy đã cài sẵn môi trường) |

`run_all.sh` tự `cd` vào thư mục chứa nó nên đặt repo ở đâu cũng chạy được.

## 5. Kết quả

Trong `results/` (hoặc `$OUT`):

- `run_all_log.txt`: toàn bộ output của cả phiên chạy.
- `<tên bài>.log`: log đầy đủ từng bài do `resource_monitor.py` ghi (stdout + stderr).
- `<tên bài>.console.log`: output của tiến trình `resource_monitor.py` (dùng khi chạy song song).
- `<tên bài>_samples.csv`: CPU, RAM, GPU mỗi giây. `<tên bài>_summary.txt`: tổng kết tài nguyên.
- `metrics_log.txt`: sau mỗi bài, bảng tp/fp/fn/precision/recall/fscore và thời gian chạy.
- `results_summary.csv`: bảng metric gộp.
- `verify.txt`: kết quả `verify_final.sh`.
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
| `pg_restore: error: ... already exists` | DB nạp dở trước đó. `dropdb --force <db>` rồi nạp lại, hoặc `RESTORE=force ./run_all.sh`. |
| `SSL error: unexpected eof` hoặc `Input/output error` lúc nạp | Hết đĩa. Giải phóng đĩa (với WSL là ổ C:), `dropdb --force` DB dở rồi nạp lại. |
| `database ... is being accessed by other users` | Có kết nối treo. `SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname='<db>';` rồi `dropdb --force`. |
| `pg_restore` từ chối dump | Dump tạo bằng Postgres mới hơn bản đang cài. Dùng Postgres 17 hoặc dump lại bằng đúng phiên bản. |
| Bài B báo lỗi ở bước dựng graph của `TRACE_E3` | Ngày test của `TRACE_E3` trống. Bỏ các ngày đó khỏi config dataset `TRACE_E3`. |
| `torch.cuda.is_available()` ra `False` | Driver chưa đúng hoặc GPU không được hỗ trợ. Kiểm tra `nvidia-smi` và phiên bản CUDA. |
| `collect_results loi` trong `metrics_log.txt` | Chạy trong đúng môi trường conda `pids` (cần torch). |

## 8. Cấu trúc

```
PIDSMaker_baselines.md         tài liệu này (hướng dẫn + toàn bộ mã nguồn)
run_all.sh                     chạy 4 bài, kiểm tra/nạp DB, log, thông báo, dọn đĩa
run_parallel.sh                wrapper: chạy run_all.sh ở chế độ song song
resource_monitor.py            đo CPU / RAM / GPU quanh mỗi lệnh
collect_results.py             gom metric từ artifacts
prepare_cutoff_db.py           tạo DB cắt theo cutoff (bài A)
scrub_inplace.py               xóa node attack và event nối vào chúng, tại chỗ
check_train_contamination.py   kiểm tra node attack có nằm trong ngày train/val không
make_test_gt.py                tạo ground truth cho phần test
verify_final.sh                kiểm tra lần cuối các DB
vm_setup.sh                    cài đặt môi trường
download_datasets.sh           tải dump từ Google Drive
tests/                         test cross-host
pidsmaker/, config/            mã PIDSMaker và cấu hình
Ground_Truth/                  ground truth (gồm bản sau cutoff cho bài A)
final_checksums.txt            sha256 của các file trong gói
```

## 9. Giới hạn đã biết

- Kết quả kiểm tra nhiễm chỉ dựa vào các attack đã được gán nhãn trong file GT. Hoạt động độc hại chưa được gán nhãn thì không phát hiện được.
- Xóa node GT cũng xóa mọi event benign nối vào node đó (ví dụ file hệ thống dùng chung, hoặc netflow placeholder ở THEIA), nên baseline mất một phần hành vi bình thường.
- `trace_e3` chỉ dùng để train. Không đánh giá attack trên `trace_e3` và `theia_e3` đã xóa node.

---

# 10. Chạy tất cả trong MỘT ô bash

Chỉ cần copy **một** ô dưới đây vào terminal (đứng trong thư mục repo) và chạy. Ô này:

1. Ghi và chạy `vm_setup.sh` **đầu tiên** (cài Postgres, conda, torch, thư viện). Đặt `SKIP_SETUP=1` nếu máy đã cài sẵn.
2. Ghi toàn bộ script cần thiết và áp patch lõi PIDSMaker.
3. Nạp DB còn thiếu, tạo index, chạy 4 bài (tự song song nếu >= 2 GPU), gom metric, dọn đĩa, verify và gửi thông báo.

````bash
# [#1] STEP 1 - SETUP + CHECK GPU + ENV + RUN PIPELINE
set -uo pipefail

REPO="${REPO:-$(pwd)}"
cd "$REPO"
export OUT="${OUT:-results}"
export ARTIFACT_DIR="${ARTIFACT_DIR:-$HOME/artifacts}"
export DUMP_DIR="${DUMP_DIR:-data}"
export DB_HOST="${DB_HOST:-127.0.0.1}" DB_PORT="${DB_PORT:-5432}"
export PGUSER="${PGUSER:-postgres}" PGPASSWORD="${PGPASSWORD:-postgres}"
export PARALLEL="${PARALLEL:-auto}" RESTORE="${RESTORE:-missing}" VERIFY="${VERIFY:-1}"
export SEEDS="${SEEDS:-}" ONLY="${ONLY:-}"

echo "=== [#1] STEP 1: CHECK HE THONG ==="
echo "OS: $(. /etc/os-release 2>/dev/null && echo "$PRETTY_NAME" || uname -s)"
echo "Kernel: $(uname -r) | CPU: $(nproc) cores | RAM: $(free -h | awk '/^Mem:/{print $2}') | Disk($REPO): $(df -h "$REPO" | awk 'NR==2{print $4" free"}')"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || true
  nvidia-smi 2>/dev/null | grep -i "CUDA Version" || true
else
  echo "GPU: khong thay nvidia-smi (dat COMMON=--cpu de chay CPU)"
fi
command -v psql >/dev/null 2>&1 && psql --version || echo "Postgres: chua cai (vm_setup se cai)"

# Token Kaggle: doc tu bien moi truong hoac file ngoai repo (khong nhung token vao file nay)
if [ -z "${KAGGLE_API_TOKEN:-}" ] && [ -f "$HOME/.kaggle/access_token" ]; then
  export KAGGLE_API_TOKEN="$(tr -d '\r\n' < "$HOME/.kaggle/access_token")"
fi
export KAGGLE_USERNAME="${KAGGLE_USERNAME:-nguynnhtmlp11a13}"

# Ghi vm_setup.sh va chay no truoc tien (dat SKIP_SETUP=1 de bo qua)
mkdir -p "$(dirname 'vm_setup.sh')"
cat > 'vm_setup.sh' <<'__PIDSM_EOF_SETUP__'
#!/usr/bin/env bash
# vm_setup.sh - cai moi truong PIDSMaker KHONG dung Docker tren VM Ubuntu 22.04 co GPU NVIDIA.
# Lam theo dung cac phien ban duoc ghim trong Dockerfile cua repo (python 3.9, torch 2.1.2+cu121, numpy 1.26.4...).
#
# Chay 1 lan:   bash vm_setup.sh
# Bien moi truong: REPO_URL (mac dinh upstream), REPO_DIR (~/PIDSMaker), COMMIT (438a05a), CONDA_DIR (~/miniforge3)
#
# LUU Y (chua duoc chay thu tren VM that):
#  * torch 2.1.2+cu121 chay tot tren GPU dong Ampere/Hopper (A100, H100, A10, L4, RTX 30/40, ...).
#    GPU Blackwell moi (B200, RTX 50) can torch moi hon -> khong dung duoc script nay nguyen xi.
#  * driver NVIDIA phai ho tro CUDA 12.1 tro len (nvidia-smi hien CUDA Version >= 12.1).
#  * script khong dung dong sudo neu ban da la root.
set -euo pipefail

REPO_URL=${REPO_URL:-https://github.com/ubc-provenance/PIDSMaker.git}
REPO_DIR=${REPO_DIR:-$HOME/PIDSMaker}
COMMIT=${COMMIT:-438a05a}
CONDA_DIR=${CONDA_DIR:-$HOME/miniforge3}
SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"

echo "== 1/6 goi he thong =="
$SUDO apt-get update -y
$SUDO apt-get install -y --no-install-recommends git curl wget ca-certificates gnupg lsb-release graphviz postgresql-common

echo "== 2/6 PostgreSQL 17 (cung ban voi image postgres:17 cua repo) =="
if ! command -v psql >/dev/null || ! psql --version | grep -q " 17"; then
  $SUDO /usr/share/postgresql-common/pgdg/apt.postgresql.org.sh -y
  $SUDO apt-get install -y postgresql-17
fi
$SUDO pg_ctlcluster 17 main start 2>/dev/null || $SUDO service postgresql start || true
$SUDO -u postgres psql -c "ALTER USER postgres PASSWORD 'postgres';" >/dev/null
echo "   Postgres OK: $(psql --version)"

echo "== 3/6 conda (Miniforge, khong can chap nhan dieu khoan kenh defaults) =="
if [ ! -x "$CONDA_DIR/bin/conda" ]; then
  wget -q -O /tmp/miniforge.sh "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
  bash /tmp/miniforge.sh -b -p "$CONDA_DIR"
fi
# shellcheck disable=SC1091
source "$CONDA_DIR/etc/profile.d/conda.sh"
conda env list | grep -q "^pids " || conda create -y -n pids -c conda-forge --override-channels python=3.9
conda activate pids

echo "== 4/6 thu vien Python (ban ghim trong Dockerfile) =="
pip install --upgrade pip
pip install psycopg2-binary tqdm \
    scikit-learn==1.6.1 networkx==3.2.1 xxhash==3.2.0 graphviz==0.20.1 psutil scipy==1.13.1 matplotlib==3.9.4 \
    wandb==0.24.2 chardet==5.2.0 nltk==3.8.1 igraph==0.11.5 cairocffi==1.7.0 wget==3.2
pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 torchaudio==2.1.2+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
pip install torch_geometric==2.6.1 --no-cache-dir
pip install pyg_lib==0.4.0 torch_scatter==2.1.2 torch_sparse==0.6.18 torch_cluster==1.6.3 torch_spline_conv==1.2.2 \
    -f https://data.pyg.org/whl/torch-2.1.0+cu121.html --no-cache-dir
pip install gensim==4.4.0 pytz==2024.1 pandas==2.3.3 yacs==0.1.8 transformers==4.39.3
pip install numpy==1.26.4
pip install gdown==5.2.0 umap-learn==0.5.6 flask==3.0.3 pytest==8.3.5
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')" || true

echo "== 5/6 ma nguon PIDSMaker @ ${COMMIT} =="
if [ ! -d "$REPO_DIR/.git" ] && [ ! -f "$REPO_DIR/pidsmaker/main.py" ]; then git clone "$REPO_URL" "$REPO_DIR"; fi
cd "$REPO_DIR"
if [ -d .git ]; then
  git fetch --all -q || true
  git checkout -q "$COMMIT" 2>/dev/null || echo "   (khong checkout duoc $COMMIT - co the ban dang dung fork cua minh, bo qua)"
fi
pip install -e . >/dev/null 2>&1 || true

echo "== 6/6 kiem tra =="
python - <<'EOF'
import torch, numpy, gensim, torch_geometric, torch_scatter
print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available(), "| numpy", numpy.__version__)
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print("GPU:", p.name, f"{p.total_memory/2**30:.0f} GiB")
EOF
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "(khong thay nvidia-smi)"
echo
echo "XONG. Moi lan mo terminal moi:  source $CONDA_DIR/etc/profile.d/conda.sh && conda activate pids"
__PIDSM_EOF_SETUP__

if [ "${SKIP_SETUP:-0}" != "1" ]; then
  echo "=== vm_setup.sh: cai moi truong (Postgres, conda, torch, thu vien) ==="
  REPO_DIR="$REPO" bash ./vm_setup.sh || { echo "vm_setup.sh THAT BAI -> dung."; exit 1; }
fi

# Kiem tra nhanh moi truong sau setup
command -v psql >/dev/null 2>&1 || { echo 'Thieu PostgreSQL. Xem log vm_setup.sh.'; exit 1; }
PY="${PY:-}"
if [ -z "$PY" ]; then
  if [ -x "$HOME/miniforge3/envs/pids/bin/python" ]; then PY="$HOME/miniforge3/envs/pids/bin/python"
  elif [ -x "$HOME/anaconda3/envs/pids/bin/python" ]; then PY="$HOME/anaconda3/envs/pids/bin/python"
  elif command -v python >/dev/null 2>&1; then PY=python
  else echo "Thieu python. Xem log vm_setup.sh."; exit 1; fi
fi
export PY
"$PY" -c "import torch" >/dev/null 2>&1 || { echo "Khong import duoc torch trong $PY. Xem log vm_setup.sh."; exit 1; }
if ! pg_isready -h "$DB_HOST" -p "$DB_PORT" >/dev/null 2>&1; then
  sudo -n service postgresql start >/dev/null 2>&1 || sudo -n pg_ctlcluster 17 main start >/dev/null 2>&1 || true
fi

# Ghi cac script con lai
mkdir -p "$(dirname 'run_all.sh')"
cat > 'run_all.sh' <<'__PIDSM_EOF_0__'
#!/usr/bin/env bash
# =============================================================================
# run_all.sh - chay TOAN BO baseline PIDSMaker trong MOT lan, tu dau den cuoi:
#
#   - kiem tra moi truong (python/torch/postgres)
#   - chuan bi database (tu dong restore tu data/*.dump neu thieu)
#   - tao index timestamp_rec cho cac DB can dung
#   - chay 4 bai (tu dong song song neu may co >= 2 GPU)
#   - gom metric, don dep dia, verify
#
# 4 bai:
#   A_velox / A_orthrus : same-host  -> train+test THEIA_E3_CUT
#   B_velox / B_orthrus : cross-host -> train THEIA_E3 + TRACE_E3, test THEIA_E5
#
# Chay (sau khi da cai moi truong 1 lan bang vm_setup.sh):
#   bash run_all.sh
#   ONLY="A_velox" bash run_all.sh
#   SEEDS="1" bash run_all.sh          # them 4 ban seed -> lap day 8 GPU
#   PARALLEL=0 bash run_all.sh         # chay tuan tu
#   DRY_RUN=1 bash run_all.sh          # chi in lenh, khong chay
#   RESTORE=force bash run_all.sh      # drop + nap lai DB tu data/*.dump
#   VERIFY=0 bash run_all.sh           # bo qua buoc verify cuoi
#
# Bien moi truong (mac dinh trong ngoac):
#   PY (python)                 python trong conda env pids
#   OUT (results)               thu muc log + metric
#   ARTIFACT_DIR ($HOME/artifacts)  ket qua trung gian
#   DUMP_DIR (data)             noi tim <db>.dump
#   DB_HOST (127.0.0.1) DB_PORT (5432) PGUSER/PGPASSWORD (postgres/postgres)
#   NUM_GPUS (tu dong)          so GPU vat ly
#   SEEDS ()                    vd "1" hoac "1 2": them ban seed cho moi bai
#   ISOLATE (0)                 1 = ep moi bai 1 hash rieng (--restart_from_scratch)
#   COMMON ()                   tham so them cho moi bai, vd "--cpu"
#   CLEAN (after)               after = don artifacts trung gian; none = giu het
#   KEEP_RE (^(evaluation|triage|wandb)$)  thu muc duoc giu khi don
#   MIN_FREE_GB (25)            dia trong toi thieu truoc khi chay
#   SKIP_INDEX (0)              1 = khong kiem tra/tao index timestamp_rec
#   PARALLEL (auto)             auto = song song neu >= 2 GPU; 1 = luon; 0 = tuan tu
#   RESTORE (missing)           missing = chi nap DB thieu; force = drop + nap lai
#   PG_RESTORE_JOBS (4)         so job song song cua pg_restore
#   VERIFY (1)                  1 = chay verify_final.sh cuoi cung
#   DRY_RUN (0)                 1 = chi in lenh
#
# GHI CHU:
#   - Bai B train tren THEIA_E3 + TRACE_E3 va test tren THEIA_E5. Dataset o dong
#     lenh (THEIA_E5) la dataset test; train/val cua no KHONG duoc dung de hoc
#     nho `--featurization.multi_dataset_primary_test_only=True`.
#   - Sieu tham so dung bo tuned THEIA_E3 (docs/docs/tuned_systems.md), khong
#     dung bo tuned THEIA_E5 vi se ro ri thong tin test.
#   - 4 bai mac dinh co hash artifact khac nhau nen dung chung ARTIFACT_DIR an toan.
#     Ban seed cua cung config trung hash -> luon co --restart_from_scratch.
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")"
# Cho phep `python pidsmaker/main.py` import duoc package du chua `pip install -e .`
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

# ============================== CONFIG ==============================
PY=${PY:-}
if [ -z "$PY" ]; then
  if command -v python >/dev/null 2>&1; then
    PY=python
  else
    for c in "$HOME/miniforge3/envs/pids/bin/python" "$HOME/anaconda3/envs/pids/bin/python" "/opt/conda/envs/pids/bin/python"; do
      [ -x "$c" ] && { PY="$c"; break; }
    done
    PY=${PY:-python}
  fi
fi
OUT=${OUT:-results}
INTERVAL=${INTERVAL:-1}
ONLY=${ONLY:-}
COMMON=${COMMON:-}
DB_HOST=${DB_HOST:-127.0.0.1}
DB_PORT=${DB_PORT:-5432}
PGUSER=${PGUSER:-postgres}
export PGPASSWORD=${PGPASSWORD:-postgres}
ARTIFACT_DIR=${ARTIFACT_DIR:-$HOME/artifacts}
DUMP_DIR=${DUMP_DIR:-data}
GDRIVE_TOKEN=${GDRIVE_TOKEN:-}
NUM_GPUS=${NUM_GPUS:-}
SEEDS=${SEEDS:-}
ISOLATE=${ISOLATE:-0}
CLEAN=${CLEAN:-after}
KEEP_RE=${KEEP_RE:-'^(evaluation|triage|wandb)$'}
MIN_FREE_GB=${MIN_FREE_GB:-25}
SKIP_INDEX=${SKIP_INDEX:-0}
PARALLEL=${PARALLEL:-auto}
RESTORE=${RESTORE:-missing}
PG_RESTORE_JOBS=${PG_RESTORE_JOBS:-4}
VERIFY=${VERIFY:-1}
DRY_RUN=${DRY_RUN:-0}

CONN="--database_host $DB_HOST --database_port $DB_PORT --artifact_dir $ARTIFACT_DIR/"
VELOX_ARGS=${VELOX_ARGS:-"--training.encoder.dropout=0.3 --training.lr=0.001 --training.node_hid_dim=320 --training.node_out_dim=320 --training.num_epochs=12 --featurization.emb_dim=256"}
ORTHRUS_ARGS=${ORTHRUS_ARGS:-"--training.encoder.dropout=0.3 --training.lr=0.001 --training.node_hid_dim=128 --training.node_out_dim=128 --training.num_epochs=12 --featurization.emb_dim=256 --construction.time_window_size=5"}
SUBJ="--evaluation.node_evaluation.only_subject_nodes=True"
CROSS="--construction.multi_dataset=THEIA_E3,TRACE_E3 --featurization.multi_dataset_training=True --batching.multi_dataset_training=True --featurization.multi_dataset_primary_test_only=True"

ISO=""
[ "$ISOLATE" = "1" ] && ISO="--restart_from_scratch"

mkdir -p "$OUT" "$ARTIFACT_DIR" || { echo "Khong tao duoc '$OUT' hoac '$ARTIFACT_DIR'"; exit 1; }
exec > >(tee -a "$OUT/run_all_log.txt") 2>&1

DONE=0
declare -a PIDS=()
declare -a NAMES=()
declare -a JOBS=()
declare -a ALL_JOBS=()

notify() {
  local msg="$*" ts; ts=$(date '+%F %T')
  echo; echo "########## [$ts] $msg ##########"; printf '\a'
  echo "[$ts] $msg" >> "$OUT/notifications.txt"
  local short; short=$(printf '%s' "$msg" | tr -d "\"'\$\`\\\\" | tr '\n' ' ' | cut -c1-400)
  if [ -n "${NOTIFY_TELEGRAM_TOKEN:-}" ] && [ -n "${NOTIFY_TELEGRAM_CHAT:-}" ]; then
    curl -s -m 10 -X POST "https://api.telegram.org/bot${NOTIFY_TELEGRAM_TOKEN}/sendMessage" \
      -d chat_id="$NOTIFY_TELEGRAM_CHAT" --data-urlencode text="$msg" >/dev/null 2>&1 || true
  fi
  if [ -n "${NOTIFY_WEBHOOK:-}" ]; then
    curl -s -m 10 -H 'Content-Type: application/json' -d "{\"text\":\"$short\",\"content\":\"$short\"}" \
      "$NOTIFY_WEBHOOK" >/dev/null 2>&1 || true
  fi
  if [ "${NOTIFY_TOAST:-1}" = "1" ] && command -v powershell.exe >/dev/null 2>&1; then
    local ps='Add-Type -AssemblyName System.Windows.Forms; Add-Type -AssemblyName System.Drawing; $n=New-Object System.Windows.Forms.NotifyIcon; $n.Icon=[System.Drawing.SystemIcons]::Information; $n.Visible=$true; $n.ShowBalloonTip(10000,"PIDSMaker","MSG",[System.Windows.Forms.ToolTipIcon]::Info); Start-Sleep 11; $n.Dispose()'
    ps=${ps//MSG/$short}
    ( powershell.exe -NoProfile -Command "$ps" >/dev/null 2>&1 & ) || true
  fi
}
trap '[ "$DONE" = 1 ] || notify "run_all.sh DUNG truoc khi xong (loi hoac Ctrl+C). Xem $OUT/run_all_log.txt"' EXIT

free_gb() { df -BG --output=avail "$1" 2>/dev/null | tail -1 | tr -dc 0-9; }

cleanup_artifacts() {   # xoa artifacts trung gian, giu KEEP_RE; $1=force bo qua CLEAN
  [ "${1:-}" = "force" ] || [ "$CLEAN" = "after" ] || return 0
  [ -n "$ARTIFACT_DIR" ] && [ "$ARTIFACT_DIR" != "/" ] && [ -d "$ARTIFACT_DIR" ] || return 0
  local before after d base; before=$(du -sBG "$ARTIFACT_DIR" 2>/dev/null | cut -f1 | tr -dc 0-9)
  for d in "$ARTIFACT_DIR"/*/; do
    [ -d "$d" ] || continue
    base=$(basename "$d")
    [[ "$base" =~ $KEEP_RE ]] && continue
    [ "$(cd "$d" && pwd)" = "$(cd "$OUT" && pwd)" ] && continue
    rm -rf "$d"
  done
  after=$(du -sBG "$ARTIFACT_DIR" 2>/dev/null | cut -f1 | tr -dc 0-9)
  echo "[clean] artifacts: ${before:-?} GB -> ${after:-?} GB (giu: $KEEP_RE)"
}

disk_low() {   # in cac cho dang thieu dia (rong = on)
  local p fr out=""
  for p in "$ARTIFACT_DIR" /mnt/c; do
    [ -d "$p" ] || continue
    fr=$(free_gb "$p"); [ -n "$fr" ] || continue
    [ "$fr" -lt "$MIN_FREE_GB" ] && out="$out $p=${fr}GB"
  done
  echo "$out"
}

guard_disk() {
  local low; low=$(disk_low); [ -z "$low" ] && return 0
  notify "DIA THAP truoc $1 (can >= ${MIN_FREE_GB} GB):$low -> don artifacts"
  cleanup_artifacts force
  low=$(disk_low); [ -z "$low" ] && return 0
  notify "VAN THIEU DIA:$low -> DUNG. Giai phong dia roi chay lai (ONLY=\"...\")."
  return 1
}

stop_tree() {   # gui INT xuong toan bo cay tien trinh (resource_monitor -> main.py)
  local pid=$1 c
  for c in $(pgrep -P "$pid" 2>/dev/null); do stop_tree "$c"; done
  kill -INT "$pid" 2>/dev/null
}

on_int() {
  echo; echo "[!] Ctrl+C - dung tat ca bai dang chay..."
  if [ "${#PIDS[@]}" -gt 0 ]; then
    for p in "${PIDS[@]}"; do [ -n "$p" ] && stop_tree "$p"; done
  fi
  wait 2>/dev/null
  exit 130
}
trap on_int INT TERM

# ============================== kiem tra moi truong ==============================
if [ "$DRY_RUN" != "1" ]; then
  command -v psql >/dev/null 2>&1 || { echo "Khong tim thay psql. Cai PostgreSQL 17 truoc (vm_setup.sh)."; exit 1; }
  if ! command -v "$PY" >/dev/null 2>&1; then
    echo "Khong tim thay lenh python '$PY'."
    echo "  -> conda activate pids   (hoac dat PY=/duong/dan/python)"
    echo "  -> hoac cai moi truong 1 lan: bash vm_setup.sh"
    exit 1
  fi
  "$PY" -c "import torch" >/dev/null 2>&1 || {
    echo "'$PY' khong import duoc torch. Kich hoat dung env (conda activate pids) hoac chay vm_setup.sh."
    exit 1
  }
  pg_isready -h "$DB_HOST" -p "$DB_PORT" >/dev/null || {
    echo "Postgres chua chay tren $DB_HOST:$DB_PORT (sudo service postgresql start)."
    exit 1
  }
fi

# ============================== danh sach bai ==============================
# dinh dang: name|model|dataset|extra_args
ALL_JOBS+=("A_velox|velox|THEIA_E3_CUT|$ISO $VELOX_ARGS $SUBJ")
ALL_JOBS+=("A_orthrus|orthrus_non_snooped|THEIA_E3_CUT|$ISO $ORTHRUS_ARGS $SUBJ")
ALL_JOBS+=("B_velox|velox|THEIA_E5|$ISO $CROSS $VELOX_ARGS $SUBJ")
ALL_JOBS+=("B_orthrus|orthrus_non_snooped|THEIA_E5|$ISO $CROSS $ORTHRUS_ARGS $SUBJ")
for s in $SEEDS; do
  ALL_JOBS+=("A_velox_s$s|velox|THEIA_E3_CUT|--restart_from_scratch --training.seed=$s --featurization.seed=$s $VELOX_ARGS $SUBJ")
  ALL_JOBS+=("A_orthrus_s$s|orthrus_non_snooped|THEIA_E3_CUT|--restart_from_scratch --training.seed=$s --featurization.seed=$s $ORTHRUS_ARGS $SUBJ")
  ALL_JOBS+=("B_velox_s$s|velox|THEIA_E5|--restart_from_scratch --training.seed=$s --featurization.seed=$s $CROSS $VELOX_ARGS $SUBJ")
  ALL_JOBS+=("B_orthrus_s$s|orthrus_non_snooped|THEIA_E5|--restart_from_scratch --training.seed=$s --featurization.seed=$s $CROSS $ORTHRUS_ARGS $SUBJ")
done

want() { [ -z "$ONLY" ] || [[ " $ONLY " == *" $1 "* ]]; }
for job in "${ALL_JOBS[@]}"; do
  IFS='|' read -r name _ _ _ <<< "$job"
  want "$name" && JOBS+=("$job")
done
if [ "${#JOBS[@]}" -eq 0 ]; then echo "Khong co bai nao khop ONLY='$ONLY'"; exit 1; fi

# ============================== chuan bi database ==============================
PSQL=(psql -U "$PGUSER" -h "$DB_HOST" -p "$DB_PORT")
db_exists() { "${PSQL[@]}" -d postgres -tAc "SELECT 1 FROM pg_database WHERE datname='$1'" 2>/dev/null | grep -q 1; }
db_has_events() { [ "$("${PSQL[@]}" -d "$1" -tAc "SELECT count(*) FROM (SELECT 1 FROM event_table LIMIT 1) t" 2>/dev/null)" = "1" ]; }

restore_db() {   # restore_db <db> <dump>
  local db=$1 dump=$2 j=${PG_RESTORE_JOBS:-4}
  if db_exists "$db" && db_has_events "$db"; then echo "[db] $db da co"; return 0; fi
  if [ ! -s "$dump" ]; then
    local found; found=$(find . -maxdepth 3 -name "$(basename "$dump")" -size +1M 2>/dev/null | head -1)
    [ -n "$found" ] && dump=$found
  fi
  if [ ! -s "$dump" ]; then echo "[db] THIEU '$dump' (khong tim thay dump cho $db)"; return 1; fi
  if db_exists "$db"; then
    echo "[db] $db ton tai nhung rong/do -> drop de nap lai"
    dropdb -U "$PGUSER" -h "$DB_HOST" -p "$DB_PORT" --force "$db" || return 1
  fi
  echo "[db] nap $db tu $dump (pg_restore -j $j) ..."
  createdb -U "$PGUSER" -h "$DB_HOST" -p "$DB_PORT" "$db" || return 1
  if ! pg_restore -U "$PGUSER" -h "$DB_HOST" -p "$DB_PORT" -d "$db" --no-owner -j "$j" "$dump"; then
    echo "[db] pg_restore -j $j loi -> thu lai khong song song"
    dropdb -U "$PGUSER" -h "$DB_HOST" -p "$DB_PORT" --force "$db" || true
    createdb -U "$PGUSER" -h "$DB_HOST" -p "$DB_PORT" "$db" || return 1
    if ! pg_restore -U "$PGUSER" -h "$DB_HOST" -p "$DB_PORT" -d "$db" --no-owner "$dump"; then
      echo "[db] nap $db THAT BAI -> drop DB do"
      dropdb -U "$PGUSER" -h "$DB_HOST" -p "$DB_PORT" --force "$db" || true
      return 1
    fi
  fi
  db_has_events "$db" || { echo "[db] $db rong sau khi nap"; return 1; }
  echo "[db] $db san sang"
}

ensure_theia_e5() {   # co the tu tai dump neu thieu
  if db_exists theia_e5 && db_has_events theia_e5; then echo "[db] theia_e5 da co"; return 0; fi
  local dump="$DUMP_DIR/theia_e5.dump"
  if [ ! -s "$dump" ]; then
    local found; found=$(find . -maxdepth 3 -name 'theia_e5.dump' -size +1M 2>/dev/null | head -1)
    [ -n "$found" ] && dump=$found
  fi
  if [ ! -s "$dump" ]; then
    if [ -z "$GDRIVE_TOKEN" ] || [ ! -f ./download_datasets.sh ]; then
      echo "[db] KHONG co theia_e5.dump."
      echo "      Dat dump vao $DUMP_DIR/ hoac chay lai voi GDRIVE_TOKEN=<token> (can ./download_datasets.sh)."
      return 1
    fi
    mkdir -p "$DUMP_DIR"
    bash ./download_datasets.sh theia_e5 "$GDRIVE_TOKEN" || { echo "[db] tai theia_e5 that bai"; return 1; }
    dump="$DUMP_DIR/theia_e5.dump"
    if [ ! -s "$dump" ]; then
      dump=$(find . -maxdepth 3 -name 'theia_e5.dump' -size +1M 2>/dev/null | head -1)
    fi
    [ -s "${dump:-}" ] || { echo "[db] tai xong nhung khong thay theia_e5.dump"; return 1; }
  fi
  restore_db theia_e5 "$dump"
}

# DB can dung theo cac bai duoc chon
declare -A NEED_DBS=()
for job in "${JOBS[@]}"; do
  IFS='|' read -r _ _ ds _ <<< "$job"
  case "$ds" in
    THEIA_E3_CUT) NEED_DBS[theia_e3_cut]=1 ;;
    THEIA_E5)     NEED_DBS[theia_e3]=1; NEED_DBS[trace_e3]=1; NEED_DBS[theia_e5]=1 ;;
  esac
done

if [ "$DRY_RUN" != "1" ]; then
  rc=0
  for db in "${!NEED_DBS[@]}"; do
    if [ "$RESTORE" = "force" ] && db_exists "$db"; then
      echo "[db] RESTORE=force -> drop $db"
      dropdb -U "$PGUSER" -h "$DB_HOST" -p "$DB_PORT" --force "$db" || { rc=1; continue; }
    fi
    if [ "$db" = "theia_e5" ]; then
      ensure_theia_e5 || rc=1
    else
      restore_db "$db" "$DUMP_DIR/$db.dump" || rc=1
    fi
  done
  [ "$rc" = "0" ] || { echo "[db] chuan bi database THAT BAI"; exit 1; }
  echo "[db] DB can thiet deu san sang"
fi

# ============================== index timestamp_rec ==============================
index_exists() {
  [ "$("${PSQL[@]}" -d "$1" -tAc "
    SELECT 1 FROM pg_index i
    JOIN pg_class t ON t.oid = i.indrelid
    JOIN pg_attribute a ON a.attrelid = t.oid AND a.attnum = i.indkey[0]
    WHERE t.relname = 'event_table' AND i.indisvalid AND a.attname = 'timestamp_rec'
    LIMIT 1" 2>/dev/null)" = "1" ]
}

ensure_index() {
  local db=$1
  if index_exists "$db"; then echo "[db] $db: da co index timestamp_rec"; return 0; fi
  echo "[db] $db: CHUA co index timestamp_rec -> dang tao (co the mat vai phut)..."
  "${PSQL[@]}" -d "$db" -c "DROP INDEX CONCURRENTLY IF EXISTS idx_event_ts_uuid;" >/dev/null 2>&1 || true
  if ! "${PSQL[@]}" -d "$db" -c \
    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_event_ts_uuid ON event_table (timestamp_rec, event_uuid);"; then
    echo "[db] $db: tao index THAT BAI"
    return 1
  fi
  echo "[db] $db: xong index timestamp_rec"
}

if [ "$DRY_RUN" != "1" ] && [ "$SKIP_INDEX" != "1" ]; then
  for db in "${!NEED_DBS[@]}"; do
    ensure_index "$db" || { echo "[db] dung lai vi khong tao duoc index"; exit 1; }
  done
fi

# ============================== chay 4 bai ==============================
if [ "$PARALLEL" = "auto" ]; then
  if command -v nvidia-smi >/dev/null 2>&1 && [ "$(nvidia-smi -L 2>/dev/null | wc -l)" -ge 2 ]; then
    PARALLEL=1
  else
    PARALLEL=0
  fi
fi
if [ -z "${NUM_GPUS:-}" ]; then
  if command -v nvidia-smi >/dev/null 2>&1; then NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l); fi
  NUM_GPUS=${NUM_GPUS:-1}
fi
[ "${NUM_GPUS:-0}" -ge 1 ] || NUM_GPUS=1

job_cmd() {   # gan vao mang CMD
  local name=$1 model=$2 ds=$3 extra=$4
  CMD=("$PY" resource_monitor.py --name "$name" --interval "$INTERVAL" --out-dir "$OUT" \
       -- "$PY" pidsmaker/main.py "$model" "$ds" $extra $CONN $COMMON)
}

collect_job_metrics() {   # $1=name
  local name=$1
  {
    echo; echo "=================== $(date '+%F %T')  $name ==================="
    "$PY" collect_results.py --root "$ARTIFACT_DIR" --monitor-dir "$OUT" --csv "$OUT/results_summary.csv" 2>&1 \
      || echo "(collect_results loi)"
  } > "$OUT/metrics_tmp.txt"
  tee -a "$OUT/metrics_log.txt" < "$OUT/metrics_tmp.txt" >/dev/null
  tail -n 12 "$OUT/metrics_tmp.txt"
  rm -f "$OUT/metrics_tmp.txt"
}

if [ "$DRY_RUN" = "1" ]; then
  echo "[dry-run] mode=$([ "$PARALLEL" = "1" ] && echo parallel || echo sequential) | ${#JOBS[@]} bai | ${NUM_GPUS} GPU"
  idx=0
  for job in "${JOBS[@]}"; do
    IFS='|' read -r name model ds extra <<< "$job"
    job_cmd "$name" "$model" "$ds" "$extra"
    gpu=$(( idx % NUM_GPUS ))
    if [ "$PARALLEL" = "1" ]; then printf 'CUDA_VISIBLE_DEVICES=%s ' "$gpu"; fi
    printf '%q ' "${CMD[@]}"; echo
    idx=$((idx + 1))
  done
  echo "(dry-run, khong chay)"; DONE=1; exit 0
fi

if [ "$PARALLEL" = "1" ]; then
  guard_disk "ALL" || exit 3
  echo "[parallel] chay ${#JOBS[@]} bai tren ${NUM_GPUS} GPU"
  notify "[parallel] BAT DAU ${#JOBS[@]} bai tren ${NUM_GPUS} GPU (artifacts: $ARTIFACT_DIR)"
  declare -A START_TS=()
  idx=0
  for job in "${JOBS[@]}"; do
    IFS='|' read -r name model ds extra <<< "$job"
    gpu=$(( idx % NUM_GPUS ))
    job_cmd "$name" "$model" "$ds" "$extra"
    echo "[launch] $name -> GPU $gpu"
    CUDA_VISIBLE_DEVICES="$gpu" "${CMD[@]}" > "$OUT/$name.console.log" 2>&1 &
    PIDS+=("$!"); NAMES+=("$name"); START_TS["$name"]=$SECONDS
    idx=$((idx + 1))
  done

  rc_all=0
  for i in "${!PIDS[@]}"; do
    wait "${PIDS[$i]}"; rc=$?
    name=${NAMES[$i]}; dur=$((SECONDS - ${START_TS[$name]:-SECONDS}))
    echo "[done] $name exit=$rc  ($((dur/60)) phut, log: $OUT/$name.log)"
    [ "$rc" -ne 0 ] && rc_all=1
    notify "[$name] XONG exit=$rc, $((dur/60)) phut."
  done

  echo; echo "################ KET QUA ################"
  "$PY" collect_results.py --root "$ARTIFACT_DIR" --monitor-dir "$OUT" --csv "$OUT/results_summary.csv" || true
  if [ "$rc_all" -eq 0 ]; then
    cleanup_artifacts
  else
    echo "CO BAI LOI. Artifacts duoc giu lai de debug."
  fi
else
  rc_all=0
  for job in "${JOBS[@]}"; do
    IFS='|' read -r name model ds extra <<< "$job"
    guard_disk "$name" || { DONE=1; exit 3; }
    echo; echo "################ $name ################"
    notify "[$name] BAT DAU  (dia trong WSL $(free_gb "$ARTIFACT_DIR") GB, C: $(free_gb /mnt/c 2>/dev/null) GB)"
    t0=$SECONDS
    job_cmd "$name" "$model" "$ds" "$extra"
    "${CMD[@]}"
    rc=$?
    dur=$((SECONDS - t0))
    echo "[$name] exit code: $rc"
    tail_metrics=$(collect_job_metrics "$name")
    if [ "$rc" -eq 0 ]; then cleanup_artifacts; else echo "[clean] bai loi -> giu artifacts de debug"; fi
    notify "[$name] XONG exit=$rc, $((dur/60)) phut. Log: $OUT/$name.log | metric: $OUT/metrics_log.txt
$tail_metrics"
    [ "$rc" -ne 0 ] && rc_all=1
  done
  DONE=1
  [ "$rc_all" -ne 0 ] && { DONE=0; echo; echo "CO BAI LOI. Artifacts duoc giu lai de debug."; }
fi

# ============================== verify ==============================
if [ "$VERIFY" != "0" ] && [ -f ./verify_final.sh ]; then
  echo; echo "################ VERIFY ################"
  PY="$PY" DB_HOST="$DB_HOST" DB_PORT="$DB_PORT" PGUSER="$PGUSER" \
    bash ./verify_final.sh 2>&1 | tee -a "$OUT/verify.txt"
fi

echo; echo "################ KET QUA ################"
echo "Metric: $OUT/results_summary.csv | log: $OUT/run_all_log.txt | verify: $OUT/verify.txt"
DONE=1
if [ "${rc_all:-0}" -eq 0 ]; then
  notify "TAT CA XONG. Metric: $OUT/metrics_log.txt , $OUT/results_summary.csv | log tong: $OUT/run_all_log.txt"
  exit 0
else
  notify "CHAY XONG NHUNG CO BAI LOI. Xem $OUT/*.console.log va $OUT/run_all_log.txt (artifacts duoc giu lai)."
  exit 1
fi
__PIDSM_EOF_0__
mkdir -p "$(dirname 'run_parallel.sh')"
cat > 'run_parallel.sh' <<'__PIDSM_EOF_1__'
#!/usr/bin/env bash
# run_parallel.sh - giu tuong thich nguoc: chay run_all.sh o che do SONG SONG
# (moi bai 1 GPU qua CUDA_VISIBLE_DEVICES). Toan bo logic nam trong run_all.sh:
#   - tu dong restore DB tu data/*.dump neu thieu
#   - tu dong tao index timestamp_rec
#   - tu dong chon song song khi >= 2 GPU
#
# Cach dung:
#   bash run_parallel.sh                 # = PARALLEL=1 bash run_all.sh
#   SEEDS="1" bash run_parallel.sh       # + 4 ban seed (lap day 8 GPU)
#   ONLY="A_velox" bash run_parallel.sh
#   DRY_RUN=1 bash run_parallel.sh
#   CLEAN=none bash run_parallel.sh
#
# Moi bien moi truong cua run_all.sh deu duoc ho tro (xem phan CONFIG o dau run_all.sh).
export PARALLEL=${PARALLEL:-1}
export OUT=${OUT:-results_parallel}
exec bash ./run_all.sh "$@"
__PIDSM_EOF_1__
mkdir -p "$(dirname 'resource_monitor.py')"
cat > 'resource_monitor.py' <<'__PIDSM_EOF_2__'
#!/usr/bin/env python3
"""
resource_monitor.py - chay 1 lenh va do CPU / RAM / VRAM theo thoi gian, tach theo tung task cua PIDSMaker.

Cach dung (dau '--' ngan cach lenh duoc do):
  python resource_monitor.py --name A_velox --interval 1 -- \\
      python pidsmaker/main.py velox THEIA_E3_CUT <tham so...>

Dau ra trong --out-dir (mac dinh monitor_out/):
  <name>.log            toan bo stdout+stderr cua lenh (nhu khi chay binh thuong)
  <name>_samples.csv    moi dong = 1 lan lay mau (theo --interval giay)
  <name>_summary.txt    bang tong ket: toan bo chay + tung task (construction, featurization, training, ...)
  <name>_summary.json   cung noi dung, dang may doc

Do cai gi:
  proc_cpu_pct   CPU cua cay tien trinh lenh (tong cac tien trinh con), 100 = 1 nhan day, co the > 100
  machine_cpu_pct CPU toan may (0-100)
  proc_rss_gb    RAM (RSS) cua cay tien trinh - co the dem trung bo nho chia se giua cac tien trinh con
  sys_ram_used_gb RAM toan may dang dung (total - available)
  gpu_mem_used_mib  VRAM dang dung tren TAT CA GPU (nvidia-smi, gom ca tien trinh khac neu co)
  gpu_mem_proc_mib  VRAM cua rieng cay tien trinh lenh (neu nvidia-smi liet ke duoc pid; trong container
                    thuong khong liet ke duoc -> de trong)
  gpu_util_pct   do su dung GPU lon nhat trong cac GPU

Gioi han: lay mau moi --interval giay nen bo lo dinh nho ngan hon khoang do (giam --interval neu can).
Ngoai ra PIDSMaker tu log them 'peak inference ... GPU memory' (max_memory_allocated cua PyTorch) o buoc
inference - so do chinh xac hon cho rieng buoc do, con tool nay bao phu moi buoc.
"""
import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime

try:
    import psutil
except ImportError:  # pragma: no cover
    sys.exit("Thieu psutil: pip install psutil")

GB = 1024 ** 3
TASK_RE = re.compile(r"\bTask ([A-Za-z_]+): ")


# ----------------------------------------------------------------------------- GPU
class Gpu:
    def __init__(self):
        self.exe = shutil.which("nvidia-smi")
        self.ok = self.exe is not None
        self.info = []          # [{index, name, total_mib}]
        if self.ok:
            rows = self._run(["--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"])
            if rows is None:
                self.ok = False
            else:
                for r in rows:
                    p = [x.strip() for x in r.split(",")]
                    if len(p) >= 3:
                        self.info.append(dict(index=p[0], name=p[1], total_mib=_num(p[2])))
                self.ok = bool(self.info)

    def _run(self, args):
        try:
            out = subprocess.run([self.exe] + args, capture_output=True, text=True, timeout=5)
        except Exception:
            return None
        if out.returncode != 0:
            return None
        return [ln for ln in out.stdout.strip().splitlines() if ln.strip()]

    def sample(self, pids):
        """-> (mem_used_mib_all_gpus, util_max_pct, mem_mib_of_our_pids_or_None)"""
        if not self.ok:
            return None, None, None
        rows = self._run(["--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"])
        used, util = None, None
        if rows:
            vals = [[_num(x) for x in r.split(",")] for r in rows]
            mems = [v[0] for v in vals if v and v[0] is not None]
            used = sum(mems) if mems else None          # '[N/A]' tren moi GPU -> khong biet (None), KHONG phai 0
            utils = [v[1] for v in vals if len(v) > 1 and v[1] is not None]
            util = max(utils) if utils else None
        mine = None
        apps = self._run(["--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"])
        if apps:
            tot, seen = 0.0, False
            for r in apps:
                p = [x.strip() for x in r.split(",")]
                if len(p) >= 2 and p[0].isdigit() and int(p[0]) in pids:
                    v = _num(p[1])
                    if v is not None:                    # '[Not Supported]' / '[N/A]' -> bo qua, khong tinh la 0
                        tot += v
                        seen = True
            mine = tot if seen else None
        return used, util, mine


def _num(x):
    try:
        return float(str(x).strip())
    except Exception:
        return None


# ------------------------------------------------------------------------ sampler
class Sampler(threading.Thread):
    FIELDS = ["t_s", "task", "proc_cpu_pct", "machine_cpu_pct", "proc_rss_gb", "sys_ram_used_gb",
              "gpu_mem_used_mib", "gpu_mem_proc_mib", "gpu_util_pct"]

    def __init__(self, root_pid, interval, csv_path, gpu, state):
        super().__init__(daemon=True)
        self.root_pid, self.interval, self.gpu, self.state = root_pid, interval, gpu, state
        self.stop_evt = threading.Event()
        self.rows = []
        self.ncpu = psutil.cpu_count(logical=True) or 1
        self._fh = open(csv_path, "w", newline="")
        self._w = csv.writer(self._fh)
        self._w.writerow(self.FIELDS)
        self.procs = {}

    def run(self):
        t0 = self.state["t0"]
        psutil.cpu_percent(interval=None)  # prime machine-wide counter
        while not self.stop_evt.is_set():
            self._sample(t0)
            self.stop_evt.wait(self.interval)
        self._sample(t0)
        self._fh.close()

    def _tree(self):
        try:
            root = psutil.Process(self.root_pid)
            kids = root.children(recursive=True)
        except psutil.Error:
            return []
        procs = []
        for p in [root] + kids:
            if p.pid not in self.procs:
                self.procs[p.pid] = p
                try:
                    p.cpu_percent(interval=None)          # prime
                except psutil.Error:
                    pass
            procs.append(self.procs[p.pid])
        return procs

    def _sample(self, t0):
        cpu, rss, pids = 0.0, 0, set()
        for p in self._tree():
            try:
                cpu += p.cpu_percent(interval=None)
                rss += p.memory_info().rss
                pids.add(p.pid)
            except psutil.Error:
                continue
        vm = psutil.virtual_memory()
        g_used, g_util, g_mine = self.gpu.sample(pids)
        row = [round(time.time() - t0, 2), self.state["task"], round(cpu, 1),
               round(psutil.cpu_percent(interval=None), 1), round(rss / GB, 3),
               round((vm.total - vm.available) / GB, 3),
               "" if g_used is None else g_used, "" if g_mine is None else g_mine,
               "" if g_util is None else g_util]
        self.rows.append(row)
        self._w.writerow(row)
        self._fh.flush()


# ------------------------------------------------------------------------ summary
def _agg(rows, col, fn):
    i = Sampler.FIELDS.index(col)
    vals = [r[i] for r in rows if r[i] != ""]
    return fn(vals) if vals else None


def summarize(rows, spans, total_s, exit_code, gpu, ncpu):
    def block(sub, dur):
        return dict(
            duration_s=round(dur, 1), samples=len(sub),
            cpu_proc_avg_pct=_r(_agg(sub, "proc_cpu_pct", lambda v: sum(v) / len(v))),
            cpu_proc_peak_pct=_r(_agg(sub, "proc_cpu_pct", max)),
            cpu_machine_avg_pct=_r(_agg(sub, "machine_cpu_pct", lambda v: sum(v) / len(v))),
            rss_peak_gb=_r(_agg(sub, "proc_rss_gb", max), 2),
            sys_ram_peak_gb=_r(_agg(sub, "sys_ram_used_gb", max), 2),
            gpu_mem_peak_mib=_r(_agg(sub, "gpu_mem_used_mib", max), 0),
            gpu_mem_proc_peak_mib=_r(_agg(sub, "gpu_mem_proc_mib", max), 0),
            gpu_util_avg_pct=_r(_agg(sub, "gpu_util_pct", lambda v: sum(v) / len(v))),
            gpu_util_peak_pct=_r(_agg(sub, "gpu_util_pct", max)),
        )

    out = dict(
        exit_code=exit_code, total_s=round(total_s, 1), cpu_cores=ncpu,
        machine_ram_gb=round(psutil.virtual_memory().total / GB, 1),
        gpus=gpu.info if gpu.ok else "none (nvidia-smi khong co hoac khong chay duoc)",
        overall=block(rows, total_s), tasks={},
    )
    order = []
    for name, a, b in spans:
        if name not in order:
            order.append(name)
    for name in order:
        parts = [(a, b) for n, a, b in spans if n == name]
        sub = [r for r in rows if any(a <= r[0] <= b for a, b in parts)]
        out["tasks"][name] = block(sub, sum(b - a for a, b in parts))
    return out


def _r(x, nd=1):
    return None if x is None else round(x, nd)


def render(s):
    def f(v, unit=""):
        return "-" if v is None else f"{v}{unit}"
    L = []
    L.append(f"exit code: {s['exit_code']} | thoi gian: {s['total_s']} s | CPU: {s['cpu_cores']} nhan | RAM may: {s['machine_ram_gb']} GB")
    g = s["gpus"]
    L.append("GPU: " + (", ".join(f"#{x['index']} {x['name']} ({x['total_mib']:.0f} MiB)" for x in g) if isinstance(g, list) else g))
    L.append("")
    hdr = f"{'task':<16}{'giay':>8}{'CPU tb%':>9}{'CPU max%':>10}{'RSS max GB':>12}{'RAM may GB':>12}{'VRAM max MiB':>14}{'VRAM proc':>11}{'GPU tb%':>9}"
    L.append(hdr)
    L.append("-" * len(hdr))
    def row(name, b):
        return (f"{name:<16}{f(b['duration_s']):>8}{f(b['cpu_proc_avg_pct']):>9}{f(b['cpu_proc_peak_pct']):>10}"
                f"{f(b['rss_peak_gb']):>12}{f(b['sys_ram_peak_gb']):>12}{f(b['gpu_mem_peak_mib']):>14}"
                f"{f(b['gpu_mem_proc_peak_mib']):>11}{f(b['gpu_util_avg_pct']):>9}")
    for name, b in s["tasks"].items():
        L.append(row(name, b))
    L.append("-" * len(hdr))
    L.append(row("TONG", s["overall"]))
    L.append("")
    L.append("CPU: 100% = 1 nhan day (cay tien trinh co the > 100%). RSS co the dem trung bo nho chia se giua tien trinh con.")
    return "\n".join(L)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--out-dir", default="monitor_out")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    if not cmd:
        ap.error("thieu lenh sau dau '--'")
    os.makedirs(a.out_dir, exist_ok=True)
    base = os.path.join(a.out_dir, a.name)

    gpu = Gpu()
    state = dict(t0=time.time(), task="(start)")
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    child = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
    sampler = Sampler(child.pid, a.interval, base + "_samples.csv", gpu, state)
    sampler.start()

    spans, cur_name, cur_start = [], None, None
    log = open(base + ".log", "w", encoding="utf-8")
    try:
        for line in child.stdout:
            sys.stdout.write(line)
            log.write(line)
            m = TASK_RE.search(line)
            if m:
                now = time.time() - state["t0"]
                if not spans and cur_name is None:          # thoi gian khoi dong truoc task dau tien (nap thu vien...)
                    spans.append(("(start)", 0.0, now))
                if cur_name is not None:
                    spans.append((cur_name, cur_start, now))
                if m.group(1) == "done":                 # "Task done: <path>" = ket thuc task hien tai
                    cur_name, state["task"] = None, "(between tasks)"
                else:
                    cur_name, cur_start, state["task"] = m.group(1), now, m.group(1)
        child.wait()
    except KeyboardInterrupt:
        child.terminate()
        child.wait()
    finally:
        total = time.time() - state["t0"]
        if cur_name is not None:
            spans.append((cur_name, cur_start, total))
        sampler.stop_evt.set()
        sampler.join(timeout=5)
        log.close()

    summary = summarize(sampler.rows, spans, total, child.returncode, gpu, sampler.ncpu)
    txt = render(summary)
    with open(base + "_summary.txt", "w", encoding="utf-8") as fh:
        fh.write(txt + "\n")
    with open(base + "_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print("\n===== RESOURCE SUMMARY: " + a.name + " =====\n" + txt)
    print(f"(chi tiet: {base}_samples.csv, {base}_summary.json)")
    return child.returncode


if __name__ == "__main__":
    sys.exit(main())
__PIDSM_EOF_2__
mkdir -p "$(dirname 'collect_results.py')"
cat > 'collect_results.py' <<'__PIDSM_EOF_3__'
#!/usr/bin/env python3
"""
collect_results.py - gom ket qua danh gia (stats_*.pth) va do tai nguyen (resource_monitor) thanh 1 bang.

  python collect_results.py [--root .] [--monitor-dir results] [--csv results_summary.csv]

Can torch (chay trong moi truong PIDSMaker). --root la thu muc chua artifacts (mac dinh: tim trong thu muc hien tai,
./artifacts va --artifact_dir ban da dung).

Kiem tra quan trong: tp + fn phai bang so node GT (vd 51 process cho THEIA_E3_CUT, 64 cho THEIA_E5).
"""
import argparse
import csv
import glob
import json
import os
import sys

try:
    import torch
except ImportError:
    sys.exit("Thieu torch: chay trong moi truong PIDSMaker")

KEYS = ["tp", "fp", "fn", "tn", "precision", "recall", "fscore", "mcc", "adp_score", "percent_detected_attacks", "threshold"]


def load_stats(path):
    try:
        return torch.load(path, weights_only=False)
    except TypeError:                       # torch cu khong co weights_only
        return torch.load(path)


def dataset_of(path):
    parts = path.split(os.sep)
    if "evaluation" in parts:
        i = parts.index("evaluation")
        if len(parts) > i + 3:
            return parts[i + 3]
    return "?"


def fmt(v):
    if v is None:
        return "-"
    try:
        v = float(v)
    except Exception:
        return str(v)
    return f"{v:.3f}" if abs(v) < 1000 and v != int(v) else f"{int(v)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", action="append", default=[])
    ap.add_argument("--monitor-dir", default="results")
    ap.add_argument("--csv", default="results_summary.csv")
    a = ap.parse_args()
    roots = a.root or [".", "artifacts"]
    files = set()
    for r in roots:
        files.update(glob.glob(os.path.join(r, "**", "stats_*.pth"), recursive=True))
    rows = []
    for p in sorted(files, key=os.path.getmtime):
        s = load_stats(p)
        d = {k: s.get(k) for k in KEYS}
        d["dataset"], d["file"] = dataset_of(p), p
        tp, fn = d["tp"], d["fn"]
        d["tp+fn"] = None if tp is None or fn is None else int(tp) + int(fn)
        rows.append(d)

    print("== Ket qua danh gia (cu -> moi, theo thoi gian ghi file) ==")
    hdr = ["dataset", "tp", "fp", "fn", "tn", "tp+fn", "precision", "recall", "fscore", "mcc", "adp_score", "percent_detected_attacks"]
    print("  ".join(f"{h:>12}" for h in hdr))
    for d in rows:
        print("  ".join(f"{(d[h] if h == 'dataset' else fmt(d.get(h))):>12}" for h in hdr))
    if not rows:
        print("(khong tim thay stats_*.pth - da chay xong buoc evaluation chua? thu them --root <artifact_dir>)")

    mons = []
    for p in sorted(glob.glob(os.path.join(a.monitor_dir, "*_summary.json"))):
        m = json.load(open(p))
        o = m["overall"]
        mons.append(dict(run=os.path.basename(p)[:-len("_summary.json")], exit=m["exit_code"], seconds=m["total_s"],
                         cpu_peak_pct=o["cpu_proc_peak_pct"], rss_peak_gb=o["rss_peak_gb"], ram_sys_peak_gb=o["sys_ram_peak_gb"],
                         vram_peak_mib=o["gpu_mem_peak_mib"], vram_proc_peak_mib=o["gpu_mem_proc_peak_mib"]))
    if mons:
        print("\n== Tai nguyen theo tung lan chay (chi tiet tung task: <run>_summary.txt) ==")
        h2 = ["run", "exit", "seconds", "cpu_peak_pct", "rss_peak_gb", "ram_sys_peak_gb", "vram_peak_mib", "vram_proc_peak_mib"]
        print("  ".join(f"{h:>18}" for h in h2))
        for m in mons:
            print("  ".join(f"{(m[h] if h == 'run' else fmt(m[h])):>18}" for h in h2))

    with open(a.csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["section"] + hdr + ["file"])
        for d in rows:
            w.writerow(["eval"] + [d.get(h) for h in hdr] + [d["file"]])
        if mons:
            w.writerow([])
            w.writerow(["section"] + list(mons[0].keys()))
            for m in mons:
                w.writerow(["resource"] + list(m.values()))
    print(f"\n(da ghi {a.csv})")


if __name__ == "__main__":
    main()
__PIDSM_EOF_3__
mkdir -p "$(dirname 'prepare_cutoff_db.py')"
cat > 'prepare_cutoff_db.py' <<'__PIDSM_EOF_4__'
#!/usr/bin/env python3
"""
prepare_cutoff_db.py
====================
Tao ban sao DB Postgres cua PIDSMaker (vd theia_e3) de chay baseline (Velox, Orthrus)
cung moc cut-off voi [SYS], sach attack o phan train/val.

Lam gi:
  1. Phan tich GT: moi node GT (process/file/netflow) nam TRUOC / SAU / VAT QUA cutoff.
     --report-only chi in bao cao nay, khong dung gi vao DB.
  2. CREATE DATABASE <dst> TEMPLATE <src>   (ban sao, DB goc khong bi sua)
  3. Trong ban sao, chi voi event co timestamp < cutoff:
       a. xoa moi event cham toi node attack (src hoac dst)         <- "scrub"
       b. (tuy chon) xoa toan bo event trong cua so attack (+le)    <- "blackout"
       c. (mac dinh) xoa event nam truoc cutoff nhung cung ngay local voi cutoff,
          de ngay test chi con phan SAU cutoff                       <- "day-split"
     Event tu cutoff tro di KHONG bi dong vao (test van con attack de cham diem).
  4. Kiem tra lai (assert) va in bao cao.
  5. (tuy chon) --patch-pidsmaker: them dataset moi vao config.py + DARPA_TC_DATASETS.

Vi du:
  python prepare_cutoff_db.py --src-db theia_e3 --dst-db theia_e3_cut \\
    --cutoff "2018-04-12 16:35:00" --cutoff-tz UTC \\
    --attack-gt Ground_Truth/orthrus/E3-THEIA/node_Browser_Extension_Drakon_Dropper.csv \\
    --attack-gt Ground_Truth/orthrus/E3-THEIA/node_Firefox_Backdoor_Drakon_In_Memory.csv \\
    --extra-uuids ioc_33.txt \\
    --pidsmaker-dir /path/to/PIDSMaker --dataset THEIA_E3 --auto-blackout --blackout-margin-min 10 \\
    --patch-pidsmaker

Luu y: CREATE DATABASE ... TEMPLATE yeu cau KHONG co ket noi nao khac dang mo toi <src>,
va can them dung luong bang kich thuoc DB goc (THEIA_E3 ~12GB).
"""
import argparse
import ast
import copy
import json
import os
import re
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone as dt_timezone

import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_values
try:
    from zoneinfo import ZoneInfo
except ImportError:  # Python < 3.9
    ZoneInfo = None

NS = 10**9
NODE_TABLES = {
    "subject": "subject_node_table",
    "file": "file_node_table",
    "netflow": "netflow_node_table",
}


# --------------------------------------------------------------------------- time
def tz(name):
    """Tra ve tzinfo. Thu zoneinfo truoc, neu may khong co tz database (vd container ubuntu
    toi gian) thi dung pytz (container pids co san pytz)."""
    if name.upper() in ("UTC", "Z"):
        return dt_timezone.utc
    aliases = [name] + (["America/New_York"] if name == "US/Eastern" else [])
    if ZoneInfo is not None:
        for n in aliases:
            try:
                return ZoneInfo(n)
            except Exception:
                pass
    try:
        import pytz
        return pytz.timezone(name)
    except Exception:
        raise SystemExit(f"Khong tim thay mui gio {name!r}. Cai: pip install tzdata   (hoac pytz)")


def localize(naive, z):
    """Gan mui gio vao datetime 'naive'; pytz can .localize(), zoneinfo thi dung replace()."""
    if hasattr(z, "localize"):
        return z.localize(naive)
    return naive.replace(tzinfo=z)


def parse_dt(s, tzname):
    s = s.strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return localize(datetime.strptime(s, fmt), tz(tzname))
        except ValueError:
            pass
    raise ValueError(f"Khong doc duoc thoi gian: {s!r} (dung 'YYYY-MM-DD HH:MM[:SS]')")


def to_ns(s, tzname):
    return int(parse_dt(s, tzname).timestamp()) * NS


def fmt_utc(ns):
    if ns is None:
        return "-"
    return datetime.fromtimestamp(ns // NS, tz=dt_timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def local_midnight_ns(ns, tzname):
    z = tz(tzname)
    d = datetime.fromtimestamp(ns // NS, tz=z)
    return int(localize(datetime(d.year, d.month, d.day), z).timestamp()) * NS


# ------------------------------------------------------------------- GT / uuids
def read_gt_csv(path):
    """Dinh dang PIDSMaker: UUID,{'subject': '...'},index_id  (label co the chua dau phay)."""
    rows = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "," not in line:
                rows.append(dict(uuid=line, ntype=None, csv_idx=None, src=os.path.basename(path)))
                continue
            uuid, rest = line.split(",", 1)
            m = re.match(r"\{'(\w+)'", rest.strip())
            ntype = m.group(1) if m else None
            idx = None
            if "," in rest:
                tail = rest.rsplit(",", 1)[1].strip()
                if tail.isdigit():
                    idx = tail
            rows.append(dict(uuid=uuid.strip(), ntype=ntype, csv_idx=idx, src=os.path.basename(path)))
    return rows


def read_uuid_file(path):
    """Mot UUID moi dong (hoac CSV, cot dau la UUID). '#' la comment."""
    rows = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append(dict(uuid=line.split(",")[0].strip(), ntype=None, csv_idx=None,
                             src=os.path.basename(path)))
    return rows


def resolve_uuids(cur, uuids):
    """uuid -> {'ids': set(index_id), 'types': set(ntype)} tu 3 bang node cua DB."""
    wanted = set()
    for u in uuids:
        wanted.add(u)
        wanted.add(u.upper())
    wanted = list(wanted)
    out = {}
    for ntype, table in NODE_TABLES.items():
        cur.execute(
            sql.SQL("SELECT node_uuid, index_id FROM {} WHERE node_uuid = ANY(%s)").format(
                sql.Identifier(table)),
            (wanted,),
        )
        for u, i in cur.fetchall():
            d = out.setdefault(u, {"ids": set(), "types": set()})
            d["ids"].add(str(i))
            d["types"].add(ntype)
    return out


def build_attack_nodes(cur, rows):
    resolved = resolve_uuids(cur, {r["uuid"] for r in rows})
    nodes, missing, mismatch = [], [], []
    for r in rows:
        d = resolved.get(r["uuid"]) or resolved.get(r["uuid"].upper())
        if d:
            ids = set(d["ids"])
            ntype = r["ntype"] or sorted(d["types"])[0]
            if r["csv_idx"] and r["csv_idx"] not in ids:
                mismatch.append((r["uuid"], r["csv_idx"], sorted(ids)))
        elif r["csv_idx"]:
            ids, ntype = {r["csv_idx"]}, r["ntype"]
            missing.append(r["uuid"])
        else:
            missing.append(r["uuid"])
            continue
        nodes.append(dict(uuid=r["uuid"], ntype=ntype, src=r["src"], ids=ids))
    return nodes, missing, mismatch


def node_activity(cur, idx_list, cut_ns):
    cur.execute(
        """
        SELECT idx, MIN(ts), MAX(ts), COUNT(*), COUNT(*) FILTER (WHERE ts < %(cut)s) FROM (
            SELECT src_index_id AS idx, timestamp_rec AS ts FROM event_table
              WHERE src_index_id = ANY(%(ids)s)
            UNION ALL
            SELECT dst_index_id, timestamp_rec FROM event_table
              WHERE dst_index_id = ANY(%(ids)s)
        ) t GROUP BY idx
        """,
        {"ids": idx_list, "cut": cut_ns},
    )
    return {r[0]: (r[1], r[2], r[3], r[4]) for r in cur.fetchall()}


def classify(nodes, activity, cut_ns):
    for n in nodes:
        acts = [activity[i] for i in n["ids"] if i in activity]
        if not acts:
            n.update(first=None, last=None, n_events=0, n_before=0, n_after=0, side="no_events")
            continue
        first = min(a[0] for a in acts)
        last = max(a[1] for a in acts)
        side = "before" if last < cut_ns else ("after" if first >= cut_ns else "spanning")
        tot, nb = sum(a[2] for a in acts), sum(a[3] for a in acts)
        n.update(first=first, last=last, n_events=tot, n_before=nb, n_after=tot - nb, side=side)


def make_report(nodes, cut_ns):
    sides = ("before", "after", "spanning", "no_events")
    table = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    seen = set()
    uniq = []
    for n in nodes:
        key = (n["src"], n["uuid"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(n)
        table[n["src"]][n["ntype"] or "unknown"][n["side"]] += 1
    lines = [f"Cutoff (UTC): {fmt_utc(cut_ns)}", ""]
    hdr = f"{'file / loai node':<62}" + "".join(f"{s:>11}" for s in sides) + f"{'tong':>8}"
    lines.append(hdr)
    lines.append("-" * len(hdr))
    totals = defaultdict(int)
    subj = defaultdict(int)
    for src in sorted(table):
        for nt in sorted(table[src]):
            c = table[src][nt]
            lines.append(f"{src[:40] + ' / ' + nt:<62}" + "".join(f"{c[s]:>11}" for s in sides)
                         + f"{sum(c.values()):>8}")
            for s in sides:
                totals[s] += c[s]
                if nt == "subject":
                    subj[s] += c[s]
    lines.append("-" * len(hdr))
    lines.append(f"{'TONG (moi loai node)':<62}" + "".join(f"{totals[s]:>11}" for s in sides)
                 + f"{sum(totals.values()):>8}")
    lines.append(f"{'TONG chi process (subject)':<62}" + "".join(f"{subj[s]:>11}" for s in sides)
                 + f"{sum(subj.values()):>8}")
    uu = {n["uuid"] for n in uniq}
    uu_subj = {n["uuid"] for n in uniq if n["ntype"] == "subject"}
    lines.append(f"{'UUID duy nhat (tat ca / chi subject)':<62}{len(uu):>11}{len(uu_subj):>11}")
    spanning = [n for n in uniq if n["side"] == "spanning"]
    if spanning:
        lines += ["", f"Node VAT QUA cutoff ({len(spanning)}): co event ca truoc lan sau cutoff. "
                      "Phan truoc cutoff se bi scrub; phan sau cutoff van o trong test.",
                  "  (file / loai / uuid / so event truoc | sau cutoff / khoang thoi gian UTC)"]
        for n in sorted(spanning, key=lambda x: (x["src"], x["ntype"] or "", x["uuid"]))[:40]:
            lines.append(f"  {n['src'][:34]:<34} {(n['ntype'] or '?'):<8} {n['uuid']}  "
                         f"{n['n_before']:>8,} | {n['n_after']:<8,}  "
                         f"{fmt_utc(n['first'])} -> {fmt_utc(n['last'])}")
        if len(spanning) > 40:
            lines.append(f"  ... va {len(spanning) - 40} node nua (xem --report-json)")
    uuid_src, idx_uuid = defaultdict(set), defaultdict(set)
    for n in nodes:
        uuid_src[n["uuid"]].add(n["src"])
        for i in n["ids"]:
            idx_uuid[i].add(n["uuid"])
    dup_uuid = {u: sorted(v) for u, v in uuid_src.items() if len(v) > 1}
    shared_idx = {i: sorted(v) for i, v in idx_uuid.items() if len(v) > 1}
    if dup_uuid:
        lines += ["", f"[LUU Y] {len(dup_uuid)} UUID xuat hien o nhieu file (vd GT va IOC trung nhau):"]
        lines += [f"  {u}  <- {v}" for u, v in list(dup_uuid.items())[:10]]
    if shared_idx:
        lines += ["", f"[LUU Y] {len(shared_idx)} index_id duoc dung chung boi nhieu UUID:"]
        lines += [f"  index_id {i}  <- {v}" for i, v in list(shared_idx.items())[:10]]
    spanning_json = [dict(uuid=n["uuid"], ntype=n["ntype"], src=n["src"], n_before=n["n_before"],
                          n_after=n["n_after"], first=fmt_utc(n["first"]), last=fmt_utc(n["last"]))
                     for n in spanning]
    return "\n".join(lines), dict(
        spanning=spanning_json, dup_uuid=dup_uuid, shared_index_ids=shared_idx,
        cutoff_utc=fmt_utc(cut_ns),
        per_file={s: {nt: dict(c) for nt, c in d.items()} for s, d in table.items()},
        totals=dict(totals), subject_totals=dict(subj),
    )


# ------------------------------------------------------------------- PIDSMaker config
def load_dataset_entry(pm_dir, dataset):
    path = os.path.join(pm_dir, "pidsmaker", "config", "config.py")
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "DATASET_DEFAULT_CONFIG" for t in node.targets):
            for k, v in zip(node.value.keys, node.value.values):
                if isinstance(k, ast.Constant) and k.value == dataset:
                    return ast.literal_eval(v)
    raise SystemExit(f"Khong tim thay dataset {dataset!r} trong {path}")


def build_cut_entry(entry, dst_db, cut_ns, tzname, attack_days, log):
    e = copy.deepcopy(entry)
    cut_date = datetime.fromtimestamp(cut_ns // NS, tz=tz(tzname)).strftime("%Y-%m-%d")
    new_test, moved = [], []
    for d in e["test_dates"]:
        (new_test if d >= cut_date else moved).append(d)
    if attack_days == "train":
        e["train_dates"] = list(e["train_dates"]) + moved
    else:
        e["unused_dates"] = list(e["unused_dates"]) + moved
    e["test_dates"] = new_test
    keep_win = []
    for gt, s, en in e["attack_to_time_window"]:
        s_ns, e_ns = to_ns(s, tzname), to_ns(en, tzname)
        if s_ns >= cut_ns:
            keep_win.append([gt, s, en])
        elif e_ns > cut_ns:
            log(f"[CANH BAO] attack {gt} bat dau truoc cutoff nhung ket thuc sau cutoff, giu lai trong test.")
            keep_win.append([gt, s, en])
    keep_gt = {w[0] for w in keep_win}
    dropped = [g for g in e["ground_truth_relative_path"] if g not in keep_gt]
    e["ground_truth_relative_path"] = [g for g in e["ground_truth_relative_path"] if g in keep_gt]
    e["attack_to_time_window"] = keep_win
    e["database"] = dst_db
    e["database_all_file"] = dst_db
    return e, moved, dropped, cut_date


def patch_pidsmaker(pm_dir, new_name, entry, log):
    cfg_path = os.path.join(pm_dir, "pidsmaker", "config", "config.py")
    du_path = os.path.join(pm_dir, "pidsmaker", "utils", "dataset_utils.py")
    with open(cfg_path, encoding="utf-8") as f:
        txt = f.read()
    if f'"{new_name}":' in txt:
        log(f"[patch] {new_name} da co trong config.py, bo qua.")
    else:
        marker = "DATASET_DEFAULT_CONFIG = {\n"
        if marker not in txt:
            raise SystemExit("Khong tim thay 'DATASET_DEFAULT_CONFIG = {' trong config.py")
        body = json.dumps(entry, indent=4).replace("\n", "\n    ")
        txt = txt.replace(marker, marker + f'    {json.dumps(new_name)}: {body},\n', 1)
        ast.parse(txt)
        shutil.copyfile(cfg_path, cfg_path + ".bak")
        with open(cfg_path, "w", encoding="utf-8") as f:
            f.write(txt)
        log(f"[patch] da them {new_name} vao {cfg_path} (backup: .bak)")
    with open(du_path, encoding="utf-8") as f:
        du = f.read()
    if f'"{new_name}",' in du:
        log(f"[patch] {new_name} da co trong DARPA_TC_DATASETS, bo qua.")
    else:
        marker = "DARPA_TC_DATASETS = {\n"
        if marker not in du:
            raise SystemExit("Khong tim thay 'DARPA_TC_DATASETS = {' trong dataset_utils.py")
        du = du.replace(marker, marker + f'    "{new_name}",\n', 1)
        ast.parse(du)
        shutil.copyfile(du_path, du_path + ".bak")
        with open(du_path, "w", encoding="utf-8") as f:
            f.write(du)
        log(f"[patch] da them {new_name} vao DARPA_TC_DATASETS (backup: .bak)")


# ------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5432)
    ap.add_argument("--user", default="postgres")
    ap.add_argument("--password", default=os.environ.get("PGPASSWORD", ""))
    ap.add_argument("--src-db", required=True, help="DB goc, vd theia_e3")
    ap.add_argument("--dst-db", help="DB ban sao, vd theia_e3_cut (bat buoc neu khong --report-only)")
    ap.add_argument("--cutoff", required=True, help="vd '2018-04-12 16:35:00'")
    ap.add_argument("--cutoff-tz", default="UTC")
    ap.add_argument("--dataset-tz", default="US/Eastern",
                    help="mui gio cua dataset (ngay/gio trong config PIDSMaker deu la gio nay)")
    ap.add_argument("--attack-gt", action="append", default=[], help="file GT CSV (lap lai duoc)")
    ap.add_argument("--extra-uuids", action="append", default=[],
                    help="file UUID bo sung (vd 33 subject tim theo IOC), mot UUID moi dong")
    ap.add_argument("--blackout", action="append", default=[],
                    help="'START,END' theo --dataset-tz; xoa MOI event trong cua so (chi phan truoc cutoff)")
    ap.add_argument("--blackout-margin-min", type=float, default=0.0)
    ap.add_argument("--auto-blackout", action="store_true",
                    help="lay cua so tu attack_to_time_window trong config (can --pidsmaker-dir, --dataset)")
    ap.add_argument("--day-split", choices=["drop", "keep"], default="drop",
                    help="drop: xoa event truoc cutoff trong cung ngay local voi cutoff (khuyen nghi)")
    ap.add_argument("--attack-days", choices=["train", "unused"], default="train",
                    help="ngay test nam truoc cutoff (vd 2018-04-10) se chuyen sang train (da scrub) hay unused")
    ap.add_argument("--pidsmaker-dir")
    ap.add_argument("--dataset", help="ten dataset trong config PIDSMaker, vd THEIA_E3")
    ap.add_argument("--new-name", help="ten dataset moi, mac dinh <dataset>_CUT")
    ap.add_argument("--patch-pidsmaker", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--drop-existing", action="store_true")
    ap.add_argument("--report-json")
    a = ap.parse_args()

    def log(msg=""):
        print(msg, flush=True)

    if not a.report_only and not a.dst_db:
        ap.error("--dst-db la bat buoc (tru khi --report-only)")
    if (a.auto_blackout or a.patch_pidsmaker) and not (a.pidsmaker_dir and a.dataset):
        ap.error("--auto-blackout / --patch-pidsmaker can --pidsmaker-dir va --dataset")
    if not (a.attack_gt or a.extra_uuids):
        ap.error("Can it nhat mot --attack-gt hoac --extra-uuids")

    cut_ns = int(parse_dt(a.cutoff, a.cutoff_tz).timestamp()) * NS
    day_start_ns = local_midnight_ns(cut_ns, a.dataset_tz)
    rows = []
    for p in a.attack_gt:
        rows += read_gt_csv(p)
    for p in a.extra_uuids:
        rows += read_uuid_file(p)

    conn_kw = dict(host=a.host, port=a.port, user=a.user, password=a.password)

    # ---- 1. phan tich tren DB goc
    log(f"== Phan tich tren DB goc '{a.src_db}' ==")
    conn = psycopg2.connect(dbname=a.src_db, **conn_kw)
    conn.set_session(readonly=True)
    cur = conn.cursor()
    nodes, missing, mismatch = build_attack_nodes(cur, rows)
    all_ids = sorted({i for n in nodes for i in n["ids"]})
    log(f"Doc {len(rows)} dong UUID -> {len(nodes)} node, {len(all_ids)} index_id")
    if missing:
        log(f"[CANH BAO] {len(missing)} UUID khong co trong bang node cua DB (vd {missing[:3]})")
    if mismatch:
        log(f"[CANH BAO] {len(mismatch)} node co index_id trong CSV khac DB; dung index_id cua DB "
            f"(vd {mismatch[:2]})")
    activity = node_activity(cur, all_ids, cut_ns)
    classify(nodes, activity, cut_ns)
    report_txt, report = make_report(nodes, cut_ns)
    log(report_txt)
    cur.close()
    conn.close()

    result = dict(report=report, missing_uuids=missing, cutoff_ns=cut_ns,
                  n_attack_index_ids=len(all_ids))
    if a.report_only:
        if a.report_json:
            with open(a.report_json, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
        return 0

    # ---- cua so blackout
    windows = []
    for w in a.blackout:
        s, e = w.split(",")
        windows.append((to_ns(s, a.dataset_tz), to_ns(e, a.dataset_tz), f"manual {w}"))
    entry = None
    if a.pidsmaker_dir and a.dataset:
        entry = load_dataset_entry(a.pidsmaker_dir, a.dataset)
    if a.auto_blackout:
        for gt, s, e in entry["attack_to_time_window"]:
            windows.append((to_ns(s, entry["timezone"]), to_ns(e, entry["timezone"]), gt))
    margin = int(a.blackout_margin_min * 60) * NS   # chi ap dung cho cua so bat dau TRUOC cutoff

    # ---- 2. tao ban sao
    log(f"\n== Tao ban sao '{a.dst_db}' tu '{a.src_db}' ==")
    admin = psycopg2.connect(dbname="postgres", **conn_kw)
    admin.autocommit = True
    ac = admin.cursor()
    ac.execute("SELECT 1 FROM pg_database WHERE datname = %s", (a.dst_db,))
    if ac.fetchone():
        if not a.drop_existing:
            raise SystemExit(f"DB '{a.dst_db}' da ton tai. Dung --drop-existing de tao lai.")
        ac.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(a.dst_db)))
    try:
        ac.execute(sql.SQL("CREATE DATABASE {} TEMPLATE {}").format(
            sql.Identifier(a.dst_db), sql.Identifier(a.src_db)))
    except psycopg2.Error as ex:
        raise SystemExit(f"Khong tao duoc ban sao: {ex}\nDong het ket noi toi '{a.src_db}' "
                         "(PIDSMaker, psql...) roi chay lai.")
    ac.close()
    admin.close()

    # ---- 3. scrub / blackout / day-split
    conn = psycopg2.connect(dbname=a.dst_db, **conn_kw)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM event_table")
    n_total0 = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM event_table WHERE timestamp_rec < %s", (cut_ns,))
    n_pre0 = cur.fetchone()[0]
    cur.execute("CREATE TEMP TABLE attack_ids (idx TEXT PRIMARY KEY)")
    execute_values(cur, "INSERT INTO attack_ids (idx) VALUES %s", [(i,) for i in all_ids])

    cur.execute(
        """DELETE FROM event_table e WHERE e.timestamp_rec < %s
             AND (e.src_index_id IN (SELECT idx FROM attack_ids)
               OR e.dst_index_id IN (SELECT idx FROM attack_ids))""", (cut_ns,))
    n_scrub = cur.rowcount
    log(f"[scrub]     xoa {n_scrub:,} event truoc cutoff cham toi {len(all_ids)} index_id attack")

    n_black = 0
    for s, e, name in windows:
        if s >= cut_ns:   # attack nam trong khoang test: khong dong vao (ke ca le)
            log(f"[blackout]  bo qua '{name}': cua so bat dau sau cutoff ({fmt_utc(s)} UTC)")
            continue
        s, e_c = s - margin, min(e + margin, cut_ns)
        cur.execute("DELETE FROM event_table WHERE timestamp_rec >= %s AND timestamp_rec < %s", (s, e_c))
        n_black += cur.rowcount
        log(f"[blackout]  '{name}': {fmt_utc(s)} -> {fmt_utc(e_c)} UTC, xoa {cur.rowcount:,} event")

    n_day = 0
    if a.day_split == "drop":
        cur.execute("DELETE FROM event_table WHERE timestamp_rec >= %s AND timestamp_rec < %s",
                    (day_start_ns, cut_ns))
        n_day = cur.rowcount
        log(f"[day-split] xoa {n_day:,} event cua ngay local cua cutoff truoc {fmt_utc(cut_ns)} UTC "
            f"(ngay bat dau {fmt_utc(day_start_ns)} UTC)")
    conn.commit()
    conn.autocommit = True
    cur.execute("ANALYZE event_table")
    conn.autocommit = False

    # ---- 4. kiem tra (bang tam attack_ids van con trong session)
    cur.execute(
        """SELECT COUNT(*) FROM event_table e WHERE e.timestamp_rec < %s
             AND (e.src_index_id IN (SELECT idx FROM attack_ids)
               OR e.dst_index_id IN (SELECT idx FROM attack_ids))""", (cut_ns,))
    v_pre_attack = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM event_table WHERE timestamp_rec >= %s AND timestamp_rec < %s",
                (day_start_ns, cut_ns))
    v_day = cur.fetchone()[0]
    cur.execute(
        """SELECT COUNT(*) FROM event_table e WHERE e.timestamp_rec >= %s
             AND (e.src_index_id IN (SELECT idx FROM attack_ids)
               OR e.dst_index_id IN (SELECT idx FROM attack_ids))""", (cut_ns,))
    v_post_attack = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*), MIN(timestamp_rec), MAX(timestamp_rec) FROM event_table")
    n_total1, tmin, tmax = cur.fetchone()
    cur.execute("SELECT COUNT(*) FROM event_table WHERE timestamp_rec < %s", (cut_ns,))
    n_pre1 = cur.fetchone()[0]
    cur.close()
    conn.close()

    ok = v_pre_attack == 0 and (a.day_split == "keep" or v_day == 0) and v_post_attack > 0
    log("\n== Kiem tra ban sao ==")
    log(f"Event: {n_total0:,} -> {n_total1:,}   (truoc cutoff: {n_pre0:,} -> {n_pre1:,}, "
        f"sau cutoff khong doi: {n_total0 - n_pre0:,} -> {n_total1 - n_pre1:,})")
    log(f"Khoang thoi gian con lai (UTC): {fmt_utc(tmin)} -> {fmt_utc(tmax)}")
    log(f"[{'OK' if v_pre_attack == 0 else 'LOI'}] event truoc cutoff con cham node attack: {v_pre_attack}")
    if a.day_split == "drop":
        log(f"[{'OK' if v_day == 0 else 'LOI'}] event trong ngay cutoff nhung truoc cutoff: {v_day}")
    log(f"[{'OK' if v_post_attack > 0 else 'CANH BAO'}] event cham node attack tu cutoff tro di "
        f"(phai > 0 de test con attack): {v_post_attack:,}")
    result.update(events_before=n_total0, events_after=n_total1, scrubbed=n_scrub,
                  blackout=n_black, day_split=n_day, verify_ok=ok)

    # ---- 5. config PIDSMaker
    if entry is not None:
        new_name = a.new_name or f"{a.dataset}_CUT"
        new_entry, moved, dropped, cut_date = build_cut_entry(
            entry, a.dst_db, cut_ns, a.dataset_tz, a.attack_days, log)
        log(f"\n== Config dataset moi '{new_name}' ==")
        log(f"Ngay cutoff (gio dataset): {cut_date}; chuyen khoi test: {moved} -> {a.attack_days}")
        log(f"GT bi bo khoi test (attack truoc cutoff): {dropped}")
        log(f"train_dates={new_entry['train_dates']}\nval_dates={new_entry['val_dates']}\n"
            f"test_dates={new_entry['test_dates']}\nunused_dates={new_entry['unused_dates']}")
        result["new_dataset_entry"] = new_entry
        if a.patch_pidsmaker:
            patch_pidsmaker(a.pidsmaker_dir, new_name, new_entry, log)
            log(f"Chay baseline: python pidsmaker/main.py velox {new_name} ...")
        else:
            log("(them --patch-pidsmaker de tu dong ghi vao config.py va DARPA_TC_DATASETS)")

    if a.report_json:
        with open(a.report_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
__PIDSM_EOF_4__
mkdir -p "$(dirname 'scrub_inplace.py')"
cat > 'scrub_inplace.py' <<'__PIDSM_EOF_5__'
#!/usr/bin/env python3
"""
scrub_inplace.py - xoa TAI CHO (khong copy DB) cac NODE attack (file GT) + cac event noi vao chung, o MOI ngay.

Dung lai ham cua prepare_cutoff_db.py. Mac dinh chi DEM (dry-run); them --apply de xoa that.
Khong co cutoff: xoa tat ca ngay.

  python scrub_inplace.py --dataset THEIA_E3 --db theia_e3
  python scrub_inplace.py --dataset THEIA_E3 --db theia_e3 --apply
"""
import argparse
import os
import sys

import psycopg2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prepare_cutoff_db as pc  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--db", required=True)
    ap.add_argument("--pidsmaker-dir", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5432)
    ap.add_argument("--user", default="postgres")
    ap.add_argument("--password", default=os.environ.get("PGPASSWORD", "postgres"))
    ap.add_argument("--apply", action="store_true", help="xoa that (mac dinh chi dem)")
    a = ap.parse_args()

    entry = pc.load_dataset_entry(a.pidsmaker_dir, a.dataset)
    rows = []
    gt_dir = os.path.join(a.pidsmaker_dir, "Ground_Truth", "orthrus")
    for rel in entry["ground_truth_relative_path"]:
        rows += pc.read_gt_csv(os.path.join(gt_dir, rel))

    conn = psycopg2.connect(dbname=a.db, host=a.host, port=a.port, user=a.user, password=a.password)
    cur = conn.cursor()
    nodes, missing, _ = pc.build_attack_nodes(cur, rows)
    ids = sorted({i for n in nodes for i in n["ids"]})
    print(f"{a.dataset} (db={a.db}): {len({n['uuid'] for n in nodes})} node GT, {len(ids)} index_id, "
          "xoa moi ngay")
    if not ids:
        sys.exit("Khong resolve duoc node GT nao - kiem tra --db")

    where = "(src_index_id = ANY(%(ids)s) OR dst_index_id = ANY(%(ids)s))"
    p = {"ids": ids}
    cur.execute(f"SELECT COUNT(*) FROM event_table WHERE {where}", p)
    n = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM event_table")
    total = cur.fetchone()[0]
    print(f"event se xoa: {n:,} / {total:,} (cham node GT, moi ngay)")
    ids_int = [int(i) for i in ids]
    node_tables = ("subject_node_table", "file_node_table", "netflow_node_table")
    for t in node_tables:
        cur.execute(f"SELECT COUNT(*) FROM {t} WHERE index_id = ANY(%s)", (ids_int,))
        print(f"node se xoa trong {t}: {cur.fetchone()[0]}")
    if not a.apply:
        print("DRY-RUN: chua xoa gi. Them --apply de xoa that.")
        return 0

    cur.execute(f"DELETE FROM event_table WHERE {where}", p)
    print(f"da xoa {cur.rowcount:,} event")
    for t in node_tables:
        cur.execute(f"DELETE FROM {t} WHERE index_id = ANY(%s)", (ids_int,))
        print(f"da xoa {cur.rowcount} node trong {t}")
    cur.execute(f"SELECT COUNT(*) FROM event_table WHERE {where}", p)
    left = cur.fetchone()[0]
    if left:
        conn.rollback()
        sys.exit(f"Con {left} event sau khi xoa - rollback")
    conn.commit()
    conn.autocommit = True
    cur.execute("ANALYZE")
    print("xong, da commit + ANALYZE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
__PIDSM_EOF_5__
mkdir -p "$(dirname 'check_train_contamination.py')"
cat > 'check_train_contamination.py' <<'__PIDSM_EOF_6__'
#!/usr/bin/env python3
"""
check_train_contamination.py - kiem tra DU LIEU THAT co "nhiem" khong: cac node attack (theo file GT cua dataset)
co event nam trong cac ngay TRAIN / VAL khong?

Models kieu anomaly (Velox, Orthrus) hoc "binh thuong" tu cac ngay train. Neu mot node attack da hoat dong trong ngay
train/val thi hanh vi attack bi hoc nhu binh thuong (va nguong tinh tu val bi keo len). Chay tren DB thuc truoc khi train.

Cach dung (trong thu muc goc PIDSMaker, sau khi da restore DB):
  python check_train_contamination.py --dataset THEIA_E3 --db theia_e3
  python check_train_contamination.py --dataset TRACE_E3 --db trace_e3
  python check_train_contamination.py --dataset THEIA_E3_CUT --db theia_e3_cut     # bai A
Mac dinh ket noi 127.0.0.1:5432 user/password postgres. Ma thoat: 0 = sach, 2 = co node attack trong ngay train/val.

Ket qua nay chi noi ve cac attack DA duoc gan nhan trong file GT. Hoat dong doc hai chua gan nhan khong phat hien duoc.
Neu co nhiem: xoa event cua cac node do trong cac ngay train/val (xem prepare_cutoff_db.py lam mau) hoac loai ngay do.
"""
import argparse
import os
import sys

import psycopg2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prepare_cutoff_db as pc  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="ten dataset trong config PIDSMaker (THEIA_E3, TRACE_E3, THEIA_E3_CUT...)")
    ap.add_argument("--db", required=True, help="ten database Postgres")
    ap.add_argument("--pidsmaker-dir", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5432)
    ap.add_argument("--user", default="postgres")
    ap.add_argument("--password", default=os.environ.get("PGPASSWORD", "postgres"))
    ap.add_argument("--top", type=int, default=10, help="so node vi pham hien thi")
    a = ap.parse_args()

    entry = pc.load_dataset_entry(a.pidsmaker_dir, a.dataset)
    tzname = entry["timezone"]
    pg_tz = "America/New_York" if tzname == "US/Eastern" else tzname
    split = {}
    for name, dates in (("train", entry["train_dates"]), ("val", entry["val_dates"]), ("test", entry["test_dates"]),
                        ("unused", entry["unused_dates"])):
        for d in dates:
            split[d] = name

    rows = []
    gt_dir = os.path.join(a.pidsmaker_dir, "Ground_Truth", "orthrus")
    for rel in entry["ground_truth_relative_path"]:
        rows += pc.read_gt_csv(os.path.join(gt_dir, rel))
    conn = psycopg2.connect(dbname=a.db, host=a.host, port=a.port, user=a.user, password=a.password)
    conn.set_session(readonly=True)
    cur = conn.cursor()
    nodes, missing, _ = pc.build_attack_nodes(cur, rows)
    ids = sorted({i for n in nodes for i in n["ids"]})
    print(f"{a.dataset} (db={a.db}, mui gio {tzname}): {len({n['uuid'] for n in nodes})} node GT tim thay trong DB"
          f"{f', {len(missing)} UUID khong co trong DB' if missing else ''}")
    if not ids:
        sys.exit("Khong resolve duoc node GT nao - kiem tra --db co dung dataset khong")

    cur.execute(
        """SELECT (to_timestamp(timestamp_rec/1e9) AT TIME ZONE %(tz)s)::date AS d, COUNT(*)
             FROM event_table WHERE src_index_id = ANY(%(ids)s) OR dst_index_id = ANY(%(ids)s)
            GROUP BY d ORDER BY d""", {"tz": pg_tz, "ids": ids})
    per_day = {str(d): n for d, n in cur.fetchall()}
    print(f"\n{'ngay (gio dia phuong)':<22}{'chia':<8}{'event cham node GT':>20}")
    bad_days = []
    for d in sorted(set(per_day) | {k for k in split if split[k] in ("train", "val")}):
        sp = split.get(d, "(ngoai config)")
        n = per_day.get(d, 0)
        flag = ""
        if sp in ("train", "val") and n:
            flag, bad_days = "   <== NHIEM", bad_days + [d]
        if n or sp in ("train", "val"):
            print(f"{d:<22}{sp:<8}{n:>20}{flag}")

    if not bad_days:
        print("\n=> SACH: khong node attack nao (theo file GT) co event trong ngay train/val.")
        return 0
    cur.execute(
        """SELECT idx, COUNT(*), MIN(ts) FROM (
               SELECT src_index_id AS idx, timestamp_rec AS ts, (to_timestamp(timestamp_rec/1e9) AT TIME ZONE %(tz)s)::date AS d
                 FROM event_table WHERE src_index_id = ANY(%(ids)s)
               UNION ALL
               SELECT dst_index_id, timestamp_rec, (to_timestamp(timestamp_rec/1e9) AT TIME ZONE %(tz)s)::date
                 FROM event_table WHERE dst_index_id = ANY(%(ids)s)) t
            WHERE d = ANY(%(days)s::date[]) GROUP BY idx ORDER BY COUNT(*) DESC LIMIT %(top)s""",
        {"tz": pg_tz, "ids": ids, "days": bad_days, "top": a.top})
    idx2node = {i: n for n in nodes for i in n["ids"]}
    print(f"\n=> NHIEM: {sum(per_day[d] for d in bad_days)} event cua node attack trong {len(bad_days)} ngay train/val. Node vi pham nhieu nhat:")
    for idx, cnt, first in cur.fetchall():
        n = idx2node.get(idx, {})
        print(f"   index_id={idx:<10} {n.get('ntype','?'):<8} {n.get('uuid','?')}  {cnt} event, som nhat {pc.fmt_utc(first)} UTC")
    print("   Khuyen nghi: xoa cac event do khoi ngay train/val truoc khi train (baseline benign-only).")
    return 2


if __name__ == "__main__":
    sys.exit(main())
__PIDSM_EOF_6__
mkdir -p "$(dirname 'make_test_gt.py')"
cat > 'make_test_gt.py' <<'__PIDSM_EOF_7__'
#!/usr/bin/env python3
"""
make_test_gt.py
===============
Cat NOT danh sach node attack cua phan TEST theo cung moc cutoff voi du lieu.

Van de: DB da cat o 16:35 UTC, nhung GT cua THEIA_E3_CUT chi la file cua attack 12/4.
Trong khi [SYS] coi "malicious" = bat ky node GT nao co event SAU cutoff (cat theo thoi gian),
vd 2 subject cua attack 10/4 con event sau cutoff. Hai ben dem node attack khac nhau.

Script nay tao 1 file GT moi = MOI node GT (96 chinh thuc, tuy chon them IOC) co it nhat
1 event tu cutoff tro di, roi (tuy chon) sua THEIA_E3_CUT trong config.py tro vao file do.
=> baseline va [SYS] cung dung 1 dinh nghia: "node attack cua test = node GT co event sau cutoff".

Chay (trong ~/PIDSMaker, sau khi da tao DB theia_e3_cut):
  python make_test_gt.py --db theia_e3_cut --host 127.0.0.1 --port 5432 \\
    --user postgres --password postgres --cutoff "2018-04-12 16:35:00" \\
    --attack-gt Ground_Truth/orthrus/E3-THEIA/node_Browser_Extension_Drakon_Dropper.csv \\
    --attack-gt Ground_Truth/orthrus/E3-THEIA/node_Firefox_Backdoor_Drakon_In_Memory.csv \\
    --extra-uuids ioc_33.txt --include-extra \\
    --pidsmaker-dir . --dataset THEIA_E3_CUT

Bo --include-extra neu GT cua ban chi la 96 node chinh thuc.
Dung --dry-run de chi in bao cao, khong ghi file/config.
"""
import argparse
import ast
import csv
import os
import shutil
import sys
from datetime import datetime, timedelta

import psycopg2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prepare_cutoff_db as pc  # noqa: E402  (dung lai ham doc GT, resolve UUID, dem event)


def event_counts(cur, idx_list, cut_ns):
    """idx -> (so event truoc cutoff, so event tu cutoff tro di). Tu chua, khong phu thuoc
    phien ban prepare_cutoff_db.py dang nam trong thu muc."""
    cur.execute(
        """
        SELECT idx, COUNT(*) FILTER (WHERE ts < %(cut)s), COUNT(*) FILTER (WHERE ts >= %(cut)s)
        FROM (
            SELECT src_index_id AS idx, timestamp_rec AS ts FROM event_table
              WHERE src_index_id = ANY(%(ids)s)
            UNION ALL
            SELECT dst_index_id, timestamp_rec FROM event_table
              WHERE dst_index_id = ANY(%(ids)s)
        ) t GROUP BY idx
        """,
        {"ids": idx_list, "cut": cut_ns},
    )
    return {r[0]: (r[1], r[2]) for r in cur.fetchall()}


def read_raw_gt_lines(path):
    """uuid -> dong goc trong file GT chinh thuc (giu nguyen label)."""
    out = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                out[line.split(",", 1)[0].strip()] = line
    return out


def replace_entry_lists(cfg_path, dataset, gt_rel, window):
    """Sua DUNG 2 truong cua entry `dataset` trong DATASET_DEFAULT_CONFIG (theo vi tri dong AST)."""
    with open(cfg_path, encoding="utf-8") as f:
        text = f.read()
    tree = ast.parse(text)
    lines = text.split("\n")
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "DATASET_DEFAULT_CONFIG" for t in node.targets):
            for k, v in zip(node.value.keys, node.value.values):
                if isinstance(k, ast.Constant) and k.value == dataset:
                    entry = ast.literal_eval(v)
                    entry["ground_truth_relative_path"] = [gt_rel]
                    entry["attack_to_time_window"] = [[gt_rel, window[0], window[1]]]
                    import json
                    body = json.dumps(entry, indent=4).replace("\n", "\n    ")
                    new = f'    {json.dumps(dataset)}: {body},'
                    start, end = k.lineno - 1, v.end_lineno          # 0-based [start, end)
                    shutil.copyfile(cfg_path, cfg_path + ".bak")
                    lines[start:end] = new.split("\n")
                    out = "\n".join(lines)
                    ast.parse(out)
                    with open(cfg_path, "w", encoding="utf-8") as f:
                        f.write(out)
                    return entry
    raise SystemExit(f"Khong tim thay {dataset!r} trong {cfg_path}. Chay prepare_cutoff_db.py "
                     f"--patch-pidsmaker (hoac dung config.py da gui) truoc.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5432)
    ap.add_argument("--user", default="postgres")
    ap.add_argument("--password", default=os.environ.get("PGPASSWORD", ""))
    ap.add_argument("--db", required=True, help="DB da cat (theia_e3_cut) - hoac DB goc, ket qua nhu nhau")
    ap.add_argument("--cutoff", required=True)
    ap.add_argument("--cutoff-tz", default="UTC")
    ap.add_argument("--dataset-tz", default="US/Eastern")
    ap.add_argument("--attack-gt", action="append", required=True, help="file GT chinh thuc (lap lai duoc)")
    ap.add_argument("--extra-uuids", action="append", default=[])
    ap.add_argument("--include-extra", action="store_true",
                    help="tinh ca UUID bo sung (IOC) co event sau cutoff vao GT cua test")
    ap.add_argument("--pidsmaker-dir", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--gt-version", default="orthrus", help="thu muc con trong Ground_Truth/")
    ap.add_argument("--gt-subdir", default="E3-THEIA")
    ap.add_argument("--out-name", default="node_test_after_cutoff.csv")
    ap.add_argument("--dataset", help="vd THEIA_E3_CUT: sua config.py tro vao file GT moi")
    ap.add_argument("--test-end", help="mac dinh: het ngay test cuoi cung (gio dataset) trong config")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    cut_ns = int(pc.parse_dt(a.cutoff, a.cutoff_tz).timestamp()) * pc.NS
    official, raw = [], {}
    for p in a.attack_gt:
        official += pc.read_gt_csv(p)
        raw.update(read_raw_gt_lines(p))
    extra = []
    for p in a.extra_uuids:
        extra += pc.read_uuid_file(p)
    rows = official + (extra if a.include_extra else [])

    conn = psycopg2.connect(dbname=a.db, host=a.host, port=a.port, user=a.user, password=a.password)
    conn.set_session(readonly=True)
    cur = conn.cursor()
    nodes, missing, _mismatch = pc.build_attack_nodes(cur, rows)
    all_ids = sorted({i for n in nodes for i in n["ids"]})
    counts = event_counts(cur, all_ids, cut_ns)
    for n in nodes:
        nb = sum(counts.get(i, (0, 0))[0] for i in n["ids"])
        na = sum(counts.get(i, (0, 0))[1] for i in n["ids"])
        n.update(n_before=nb, n_after=na,
                 side="spanning" if (nb and na) else "after" if na else "before" if nb else "no_events")
    cur.close()
    conn.close()

    keep, seen = [], set()
    for n in nodes:
        if n["uuid"] in seen:
            continue
        seen.add(n["uuid"])
        if n["n_after"] > 0:            # co it nhat 1 event tu cutoff tro di
            keep.append(n)
    dropped = [n for n in {n["uuid"]: n for n in nodes}.values() if n["n_after"] == 0]

    def tally(ns_):
        c = {}
        for n in ns_:
            c[n["ntype"] or "?"] = c.get(n["ntype"] or "?", 0) + 1
        return c

    print(f"Cutoff (UTC): {pc.fmt_utc(cut_ns)}   |   DB: {a.db}")
    print(f"UUID dua vao (da bo trung): {len(seen)}  (GT chinh thuc: {len({r['uuid'] for r in official})}"
          f"{', + IOC' if a.include_extra else ''})")
    print(f"GIU (co event sau cutoff): {len(keep)}  theo loai: {tally(keep)}")
    spanning = [n for n in keep if n["side"] == "spanning"]
    print(f"   trong do vat qua cutoff: {len(spanning)}  theo loai: {tally(spanning)}")
    print(f"BO (chi co event truoc cutoff): {len(dropped)}  theo loai: {tally(dropped)}")
    if missing:
        print(f"[CANH BAO] {len(missing)} UUID khong co trong DB")
    subj = sum(1 for n in keep if n["ntype"] == "subject")
    print(f"=> GT cua test: {len(keep)} node ({subj} process/subject)")

    gt_dir = os.path.join(a.pidsmaker_dir, "Ground_Truth", a.gt_version, a.gt_subdir)
    out_path = os.path.join(gt_dir, a.out_name)
    rel = f"{a.gt_subdir}/{a.out_name}"
    if a.dry_run:
        print(f"[dry-run] se ghi {out_path} va sua config (neu co --dataset)")
        return 0
    os.makedirs(gt_dir, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for n in sorted(keep, key=lambda x: (x["ntype"] or "", x["uuid"])):
            if n["uuid"] in raw:
                f.write(raw[n["uuid"]] + "\n")
            else:
                idx = sorted(n["ids"])[0]
                w.writerow([n["uuid"], "{'%s': 'extra-IOC'}" % (n["ntype"] or "subject"), idx])
    print(f"Da ghi {out_path}")

    if a.dataset:
        cfg_path = os.path.join(a.pidsmaker_dir, "pidsmaker", "config", "config.py")
        with open(cfg_path, encoding="utf-8") as f:
            tree = ast.parse(f.read())
        entry = None
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == "DATASET_DEFAULT_CONFIG" for t in node.targets):
                for k, v in zip(node.value.keys, node.value.values):
                    if isinstance(k, ast.Constant) and k.value == a.dataset:
                        entry = ast.literal_eval(v)
        if entry is None:
            raise SystemExit(f"Khong thay {a.dataset!r} trong config.py")
        z = pc.tz(entry.get("timezone", a.dataset_tz))
        start_local = datetime.fromtimestamp(cut_ns // pc.NS, tz=z).strftime("%Y-%m-%d %H:%M:%S")
        if a.test_end:
            end_local = a.test_end
        else:
            last = max(datetime.strptime(d, "%Y-%m-%d") for d in entry["test_dates"])
            end_local = (last + timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
        replace_entry_lists(cfg_path, a.dataset, rel, (start_local, end_local))
        print(f"Da sua {cfg_path}: {a.dataset}.ground_truth_relative_path = ['{rel}'], "
              f"cua so = {start_local} -> {end_local} ({entry.get('timezone')}) (backup: config.py.bak)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
__PIDSM_EOF_7__
mkdir -p "$(dirname 'verify_final.sh')"
cat > 'verify_final.sh' <<'__PIDSM_EOF_8__'
#!/usr/bin/env bash
# verify_final.sh - kiem tra lan cuoi theia_e3, trace_e3, theia_e3_cut (+ theia_e5 neu co). Chi DOC, khong sua gi.
#   cd ~/PIDSMaker && bash verify_final.sh 2>&1 | tee verify_out.txt
cd "$(dirname "$0")"
PY=${PY:-}
if [ -z "$PY" ]; then
  if command -v python >/dev/null 2>&1; then
    PY=python
  else
    for c in "$HOME/miniforge3/envs/pids/bin/python" "$HOME/anaconda3/envs/pids/bin/python" "/opt/conda/envs/pids/bin/python"; do
      [ -x "$c" ] && { PY="$c"; break; }
    done
    PY=${PY:-python}
  fi
fi
export PGPASSWORD=${PGPASSWORD:-postgres}
P="-U ${PGUSER:-postgres} -h ${DB_HOST:-127.0.0.1} -p ${DB_PORT:-5432}"

echo "===== 1. DB + dung luong ====="
psql $P -d postgres -c "SELECT datname, pg_size_pretty(pg_database_size(datname)) AS size FROM pg_database WHERE datname LIKE '%e3%' OR datname LIKE '%e5%' ORDER BY 1;"

echo; echo "===== 2. So event (ky vong: theia_e3 = 29,203,665 ; trace_e3 = 25,244,673) ====="
for db in theia_e3 trace_e3 theia_e3_cut theia_e5; do
  n=$(psql $P -d $db -tAc "SELECT count(*) FROM event_table" 2>/dev/null) && echo "$db: $n event" || echo "$db: (khong co)"
done

echo; echo "===== 3. Khoang thoi gian event moi DB (UTC) ====="
for db in theia_e3 trace_e3 theia_e3_cut; do
  echo "-- $db"; psql $P -d $db -tAc "SELECT to_timestamp(min(timestamp_rec)/1e9), to_timestamp(max(timestamp_rec)/1e9) FROM event_table"
done

echo; echo "===== 4. scrub_inplace dry-run: theia_e3, trace_e3 phai con 0 event / 0 node ====="
"$PY" scrub_inplace.py --dataset THEIA_E3 --db theia_e3
"$PY" scrub_inplace.py --dataset TRACE_E3 --db trace_e3

echo; echo "===== 5. Kiem tra nhiem (theia_e3, trace_e3 phai SACH; theia_e3_cut phai SACH train/val, con attack o ngay test) ====="
"$PY" check_train_contamination.py --dataset THEIA_E3 --db theia_e3 | tail -14
"$PY" check_train_contamination.py --dataset TRACE_E3 --db trace_e3 | tail -14
"$PY" check_train_contamination.py --dataset THEIA_E3_CUT --db theia_e3_cut | tail -16

echo; echo "===== 6. run_all.sh cu phap ====="
bash -n run_all.sh && echo "run_all.sh OK"
__PIDSM_EOF_8__
mkdir -p "$(dirname 'download_datasets.sh')"
cat > 'download_datasets.sh' <<'__PIDSM_EOF_9__'
#!/usr/bin/env bash

set -e

if [ $# -ne 2 ]; then
  echo "Usage: $0 <dataset1,dataset2,...|all> <ACCESS_TOKEN>"
  exit 1
fi

DATASETS_ARG="$1"
ACCESS_TOKEN="$2"

DATA_DIR="data"
mkdir -p "$DATA_DIR"

get_id() {
  case "$1" in
    cadets_e3) echo "1DGcGBhpavNmXTnCDd_s4NWBNh2n4-6nd" ;;
    clearscope_e3) echo "1v8vsgJxv4OPFVxx8UlYvixDq8FpkIqkd" ;;
    theia_e3) echo "1p7HoH5SdMEFi0vkrEtMeG7B-hpw741p6" ;;
    theia_e5) echo "1Q0rCCj2KnJeOqJN2NcmHnDBF1sl33g9x" ;;
    clearscope_e5) echo "1zfQBbpIRIqo5EugAT5Ou0354Ii_naINC" ;;
    optc_h051) echo "1zzwge346AdAdxldykZOai5FtUrIaWc6q" ;;
    optc_h201) echo "1OSZXCQrocFSRN7wkPM02p-BqE2WmgdLD" ;;
    optc_h501) echo "1046BVjpMql1bb5WHr9yQeB6Uq6RpngbM" ;;
    cadets_e5) echo "1Xiq7w0Ofz4jZG2PVFuNqi_i0fm28kRcT" ;;
    trace_e3) echo "1xZNBbhWQO0xGVBsg6ujdPh9UMUXiUQQd" ;;
    fivedirections_e3) echo "17YHqUMbuNwP05iaOaifxvcQc2oC9pJbZ" ;;
    fivedirections_e5) echo "1EkxlbReJgMHW4TwAypPggQBA-RTRVxs4" ;;
    atlasv2_edr) echo "1t7pW7i23Ry-Z7Tpu2B-E4gezT7Wu74ma" ;;
    carbanakv2_edr) echo "1rJN9F8JkmqMAV9zUjralcFB5wykAgLnd" ;;
    *) echo "" ;;
  esac
}

download_file() {
  NAME="$1"
  ID="$2"

  echo "Downloading ${NAME}..."
  curl -L \
    -H "Authorization: Bearer ${ACCESS_TOKEN}" \
    -C - \
    "https://www.googleapis.com/drive/v3/files/${ID}?alt=media" \
    -o "${DATA_DIR}/${NAME}.dump"
}

download_trace_e5() {
  echo "Downloading trace_e5 (part aa)..."
  curl -L \
    -H "Authorization: Bearer ${ACCESS_TOKEN}" \
    -C - \
    "https://www.googleapis.com/drive/v3/files/14asyR0arrxLS4uglx8pMiR2OuoQ4emq6?alt=media" \
    -o "${DATA_DIR}/trace_e5.dump.part_aa"

  echo "Downloading trace_e5 (part ab)..."
  curl -L \
    -H "Authorization: Bearer ${ACCESS_TOKEN}" \
    -C - \
    "https://www.googleapis.com/drive/v3/files/1SOo-McGOJJqQM4I_m7xWRF-PlFN9U7Wf?alt=media" \
    -o "${DATA_DIR}/trace_e5.dump.part_ab"

  echo "Merging trace_e5..."
  cat "${DATA_DIR}/trace_e5.dump.part_aa" \
      "${DATA_DIR}/trace_e5.dump.part_ab" \
      > "${DATA_DIR}/trace_e5.dump"
}

ALL_DATASETS="cadets_e3 clearscope_e3 theia_e3 theia_e5 clearscope_e5 \
optc_h051 optc_h201 optc_h501 cadets_e5 trace_e3 \
fivedirections_e3 fivedirections_e5 trace_e5"

if [ "$DATASETS_ARG" = "all" ]; then
  SELECTED="$ALL_DATASETS"
else
  SELECTED=$(echo "$DATASETS_ARG" | tr ',' ' ')
fi

for dataset in $SELECTED; do
  if [ "$dataset" = "trace_e5" ]; then
    download_trace_e5
  else
    ID=$(get_id "$dataset")
    if [ -z "$ID" ]; then
      echo "Unknown dataset: $dataset"
      exit 1
    fi
    download_file "$dataset" "$ID"
  fi
done

echo "All requested datasets processed in ${DATA_DIR}/"
__PIDSM_EOF_9__
mkdir -p "$(dirname 'tests/test_cross_host_multi_dataset.py')"
cat > 'tests/test_cross_host_multi_dataset.py' <<'__PIDSM_EOF_10__'
"""Tests for cross-host multi-dataset mode (`featurization.multi_dataset_primary_test_only`).

Runs standalone (`python tests/test_cross_host_multi_dataset.py`) with only `torch` installed:
the functions under test are extracted from the source files with `ast`, so none of
PIDSMaker's heavy imports are needed.
"""
import ast
import os
from types import SimpleNamespace

import torch

ROOT = os.path.join(os.path.dirname(__file__), "..", "pidsmaker")


def _extract(path, names):
    src = open(os.path.join(ROOT, path), encoding="utf-8").read()
    tree = ast.parse(src)
    chunks = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            chunks.append(ast.get_source_segment(src, node))
    assert len(chunks) == len(names), f"missing {names} in {path}"
    return "\n\n".join(chunks)


def _namespace():
    from typing import Tuple

    from torch import Tensor

    ns = {"torch": torch, "Tensor": Tensor, "Tuple": Tuple, "log_tqdm": lambda it, **k: it}
    exec(_extract("tgn.py", {"LastNeighborLoader"}), ns)
    exec(_extract("utils/data_utils.py", {"compute_tgn_graphs"}), ns)
    exec(_extract("utils/utils.py", {"get_multi_datasets", "get_training_datasets"}), ns)
    return ns


class FakeBatch:
    def __init__(self, edge_index, d=4, t=3):
        self.edge_index = edge_index
        e = edge_index.shape[1]
        self.x_src, self.x_dst = torch.randn(e, d), torch.randn(e, d)
        self.node_type_src, self.node_type_dst = torch.zeros(e, t), torch.zeros(e, t)

    def to(self, *_a, **_k):
        return self


def _make_hosts(n_hosts=3, batches=3, edges_per_batch=8, nodes_per_host=12, seed=0):
    """Hosts whose node ids all start at 0 (they overlap, as in PIDSMaker DBs)."""
    g = torch.Generator().manual_seed(seed)
    hosts, gid, rows = [], 0, []
    for h in range(n_hosts):
        data_list = []
        for _ in range(batches):
            ei = torch.randint(0, nodes_per_host, (2, edges_per_batch), generator=g)
            data_list.append(FakeBatch(ei))
            for i in range(edges_per_batch):
                rows.append((ei[0, i].item(), ei[1, i].item(), gid, h))
                gid += 1
        hosts.append(data_list)
    full = SimpleNamespace(
        src=torch.tensor([r[0] for r in rows]),
        dst=torch.tensor([r[1] for r in rows]),
        msg=torch.tensor([[float(r[2])] for r in rows]),     # msg == global edge id
        t=torch.arange(len(rows)),
        edge_type=torch.zeros(len(rows), dtype=torch.long),
    )
    host_of_edge = torch.tensor([r[3] for r in rows])
    return hosts, full, host_of_edge


def _run(reset):
    ns = _namespace()
    hosts, full, host_of_edge = _make_hosts()
    cfg = SimpleNamespace(
        tgn_neighbor_n_hop=1, fix_tgn_neighbor_loader=True, fix_buggy_orthrus_TGN=True,
        insert_neighbors_before=False, tgn_neighbor_size=3, directed=False,
    )
    # datasets = [train, val, test]; each is a list of per-host data lists (here: all hosts in "train")
    datasets = [hosts, [], []]
    ns["compute_tgn_graphs"](datasets, full, None, torch.device("cpu"), 12, cfg, 4, 3,
                             reset_between_datasets=reset)
    leaks, dropped_check = 0, 0
    for h, data_list in enumerate(hosts):
        for batch in data_list:
            gids = batch.msg_tgn[:, 0].long()                 # global ids of the edges the TGN returned
            leaks += int((host_of_edge[gids] != h).sum())     # neighbors coming from another host
            # every returned edge must connect the nodes the loader says it connects
            nid = batch.n_id_tgn
            ei = batch.edge_index_tgn
            a, b = nid[ei[0]], nid[ei[1]]
            ok = ((full.src[gids] == a) & (full.dst[gids] == b)) | ((full.src[gids] == b) & (full.dst[gids] == a))
            dropped_check += int((~ok).sum())
    return leaks, dropped_check


def test_get_training_datasets():
    ns = _namespace()
    mk = lambda flag: SimpleNamespace(  # noqa: E731
        dataset=SimpleNamespace(name="THEIA_E5"),
        construction=SimpleNamespace(multi_dataset="THEIA_E3,TRACE_E3"),
        featurization=SimpleNamespace(multi_dataset_primary_test_only=flag),
    )
    assert ns["get_multi_datasets"](mk(False)) == ["THEIA_E5", "THEIA_E3", "TRACE_E3"]
    assert ns["get_training_datasets"](mk(False)) == ["THEIA_E5", "THEIA_E3", "TRACE_E3"]
    assert ns["get_training_datasets"](mk(True)) == ["THEIA_E3", "TRACE_E3"]
    cfg_none = mk(None)                                       # arg never set in YAML -> None
    assert ns["get_training_datasets"](cfg_none) == ["THEIA_E5", "THEIA_E3", "TRACE_E3"]


def test_tgn_no_leak_with_reset_and_leak_without():
    leaks_off, bad_off = _run(reset=False)
    leaks_on, bad_on = _run(reset=True)
    print(f"reset OFF: leaked edges={leaks_off}, mismatched edges={bad_off}")
    print(f"reset ON : leaked edges={leaks_on}, mismatched edges={bad_on}")
    assert leaks_off > 0                  # the problem is real: overlapping ids leak across hosts
    assert leaks_on == 0                  # fixed: neighbors only come from the current host
    assert bad_on == 0                    # and e_id -> full_data mapping stays correct (offsets)


if __name__ == "__main__":
    test_get_training_datasets(); print("PASS test_get_training_datasets")
    test_tgn_no_leak_with_reset_and_leak_without(); print("PASS test_tgn_no_leak_with_reset_and_leak_without")
__PIDSM_EOF_10__
mkdir -p "$(dirname 'Ground_Truth/orthrus/E3-THEIA/node_test_after_cutoff.csv')"
cat > 'Ground_Truth/orthrus/E3-THEIA/node_test_after_cutoff.csv' <<'__PIDSM_EOF_11__'
0100D00F-0A01-2800-0000-00006A2F0F27,{'file': '/etc/firefox/native-messaging-hosts/pass_mgr'},511272
0100D00F-0D01-2800-0000-00000A315129,{'file': '/etc/firefox/native-messaging-hosts/passwordfile.dat'},511273
0100D00F-1D12-1E00-0000-00009F8F0C12,{'file': '/var/log/xdev'},658599
0100D00F-A60F-2400-0000-0000A43DD030,{'file': '/tmp/memtrace.so'},512410
0100D00F-A66C-1800-0000-00003423B213,{'file': '/etc/ld.so.cache'},642629
0100D00F-A84B-1E00-0000-000049134A16,{'file': '/var/log/wdev'},512304
0100D00F-A84B-1E00-0000-00008C1CB31C,{'file': '/var/log/mail'},513608
0100D00F-B700-2800-0000-00002BDE0334,{'file': '/etc/firefox/native-messaging-hosts/gtcache'},512168
05000000-1F32-0000-0000-000000000000,{'file': '/dev/glx_alsa_675'},644341
80370C6E-2D96-8D2B-B0CB-500000000040,{'netflow': '128.55.12.110:38445->141.43.176.203:80'},17081
80370C6E-39C2-9299-4497-500000000040,{'netflow': '128.55.12.110:49721->146.153.68.151:80'},17140
80370C6E-4396-8D2B-B0CB-500000000040,{'netflow': '128.55.12.110:38467->141.43.176.203:80'},17139
80370C6E-64B2-A174-5848-500000000040,{'netflow': '128.55.12.110:45668->161.116.88.72:80'},74539
80370C6E-D2AA-9534-C617-500000000040,{'netflow': '128.55.12.110:43730->149.52.198.23:80'},17974
FEFFFFFF-0000-FFFF-FFFF-000000000040,{'netflow': 'LOCAL:0->NA:0'},67683
01368CB8-0200-0000-0000-000000000020,{'subject': '/etc/firefox/native-messaging-hosts/gtcache ./gtcache'},216388
033690B8-0200-0000-0000-000000000020,{'subject': '/etc/firefox/native-messaging-hosts/gtcache ./gtcache'},216390
043694B8-0200-0000-0000-000000000020,{'subject': '/etc/firefox/native-messaging-hosts/gtcache ./gtcache'},216391
283847BC-0200-0000-0000-000000000020,{'subject': '/var/log/mail /var/log/mail'},216933
293847BC-0200-0000-0000-000000000020,{'subject': '/var/log/mail /var/log/mail'},216934
3536B9B8-0200-0000-0000-000000000020,{'subject': '/etc/firefox/native-messaging-hosts/gtcache ./gtcache'},216440
3A3530B4-0200-0000-0000-000000000020,{'subject': '/home/admin/Downloads/firefox/firefox /usr/bin/firefox'},216191
3D3530B4-0200-0000-0000-000000000020,{'subject': '/home/admin/Downloads/firefox/firefox /usr/bin/firefox'},216194
3F36C5B8-0200-0000-0000-000000000020,{'subject': '/etc/firefox/native-messaging-hosts/gtcache ./gtcache'},216450
4036C6B8-0200-0000-0000-000000000020,{'subject': '/etc/firefox/native-messaging-hosts/gtcache ./gtcache'},216451
4D38FABD-0200-0000-0000-000000000020,{'subject': '/var/log/mail /var/log/mail'},216973
5336F2B8-0200-0000-0000-000000000020,{'subject': '/etc/firefox/native-messaging-hosts/gtcache ./gtcache'},216470
55361BB9-0200-0000-0000-000000000020,{'subject': '/etc/firefox/native-messaging-hosts/gtcache ./gtcache'},216472
56361CB9-0200-0000-0000-000000000020,{'subject': '/etc/firefox/native-messaging-hosts/gtcache ./gtcache'},216473
673559B4-0200-0000-0000-000000000020,{'subject': '/home/admin/Downloads/firefox/firefox /usr/bin/firefox'},216233
68355DB4-0200-0000-0000-000000000020,{'subject': '/home/admin/Downloads/firefox/firefox /usr/bin/firefox'},216234
6C3584B4-0200-0000-0000-000000000020,{'subject': '/home/admin/Downloads/firefox/firefox /usr/bin/firefox'},216238
6D3588B4-0200-0000-0000-000000000020,{'subject': '/home/admin/Downloads/firefox/firefox /usr/bin/firefox'},216239
7235B0B4-0200-0000-0000-000000000020,{'subject': '/home/admin/Downloads/firefox/firefox /usr/bin/firefox'},216244
7737F1B9-0200-0000-0000-000000000020,{'subject': '/etc/firefox/native-messaging-hosts/gtcache ./gtcache'},216759
7837F1B9-0200-0000-0000-000000000020,{'subject': '/etc/firefox/native-messaging-hosts/gtcache ./gtcache'},216760
8F3745BA-0200-0000-0000-000000000020,{'subject': '/etc/firefox/native-messaging-hosts/gtcache ./gtcache'},216783
903745BA-0200-0000-0000-000000000020,{'subject': '/etc/firefox/native-messaging-hosts/gtcache ./gtcache'},216784
9F376ABA-0200-0000-0000-000000000020,{'subject': '/etc/firefox/native-messaging-hosts/gtcache ./gtcache'},216799
A1377EBA-0200-0000-0000-000000000020,{'subject': '/etc/firefox/native-messaging-hosts/gtcache ./gtcache'},216801
B33501B6-0200-0000-0000-000000000020,{'subject': '/home/admin/Downloads/firefox/firefox /usr/bin/firefox'},216308
B43501B6-0200-0000-0000-000000000020,{'subject': '/etc/firefox/native-messaging-hosts/pass_mgr /etc/firefox/native-messaging-hosts/pass_mgr /home/admin/.mozilla/native-messaging-hosts/pass_mgr.json'},216309
B53501B6-0200-0000-0000-000000000020,{'subject': '/bin/dash /bin/sh -c ./gtcache &>/dev/null &'},216310
B73501B6-0200-0000-0000-000000000020,{'subject': '/etc/firefox/native-messaging-hosts/gtcache ./gtcache'},216312
BC353DB6-0200-0000-0000-000000000020,{'subject': '/etc/firefox/native-messaging-hosts/gtcache ./gtcache'},216317
BD353EB6-0200-0000-0000-000000000020,{'subject': '/etc/firefox/native-messaging-hosts/gtcache ./gtcache'},216318
C1357AB6-0200-0000-0000-000000000020,{'subject': '/etc/firefox/native-messaging-hosts/gtcache ./gtcache'},216322
C2357BB6-0200-0000-0000-000000000020,{'subject': '/etc/firefox/native-messaging-hosts/gtcache ./gtcache'},216323
C337B3BA-0200-0000-0000-000000000020,{'subject': '/etc/firefox/native-messaging-hosts/gtcache ./gtcache'},216833
C437B3BA-0200-0000-0000-000000000020,{'subject': '/etc/firefox/native-messaging-hosts/gtcache ./gtcache'},216834
C6359FB6-0200-0000-0000-000000000020,{'subject': '/etc/firefox/native-messaging-hosts/gtcache ./gtcache'},216327
C735BAB6-0200-0000-0000-000000000020,{'subject': '/etc/firefox/native-messaging-hosts/gtcache ./gtcache'},216328
C835BAB6-0200-0000-0000-000000000020,{'subject': '/etc/firefox/native-messaging-hosts/gtcache ./gtcache'},216329
C935BBB6-0200-0000-0000-000000000020,{'subject': '/etc/firefox/native-messaging-hosts/gtcache ./gtcache'},216330
CA35BBB6-0200-0000-0000-000000000020,{'subject': '/etc/firefox/native-messaging-hosts/gtcache ./gtcache'},216331
CA37B9BA-0200-0000-0000-000000000020,{'subject': '/etc/firefox/native-messaging-hosts/gtcache ./gtcache'},216840
CB37BFBA-0200-0000-0000-000000000020,{'subject': '/etc/firefox/native-messaging-hosts/gtcache ./gtcache'},216841
CC3499B3-0200-0000-0000-000000000020,{'subject': '/home/admin/Downloads/firefox/firefox /usr/bin/firefox'},216082
CD37C5BA-0200-0000-0000-000000000020,{'subject': '/etc/firefox/native-messaging-hosts/gtcache ./gtcache'},216843
CE37C9BA-0200-0000-0000-000000000020,{'subject': '/etc/firefox/native-messaging-hosts/gtcache ./gtcache'},216844
EF34BCB3-0200-0000-0000-000000000020,{'subject': '/home/admin/Downloads/firefox/firefox /usr/bin/firefox'},216116
F135CDB7-0200-0000-0000-000000000020,{'subject': '/etc/firefox/native-messaging-hosts/gtcache ./gtcache'},216372
F235D6B7-0200-0000-0000-000000000020,{'subject': '/etc/firefox/native-messaging-hosts/gtcache ./gtcache'},216373
F335D6B7-0200-0000-0000-000000000020,{'subject': '/etc/firefox/native-messaging-hosts/gtcache ./gtcache'},216374
FB1F9E30-0000-0000-0000-000000000020,{'subject': '/home/admin/profile /home/admin/profile'},284530
FC1F9E30-0000-0000-0000-000000000020,{'subject': '/home/admin/profile /home/admin/profile'},284531
__PIDSM_EOF_11__

# Ap patch loi PIDSMaker neu repo con sach (neu da ap roi thi bo qua)
PATCH_TMP=$(mktemp)
cat > "$PATCH_TMP" <<'__PIDSM_PATCH__'
diff --git a/config/default.yml b/config/default.yml
index a1c974b..11f4d6b 100644
--- a/config/default.yml
+++ b/config/default.yml
@@ -189,6 +189,7 @@ evaluation:
     use_dst_node_loss: True
     use_kmeans: True
     kmeans_top_K: 30
+    only_subject_nodes: False
   node_tw_evaluation:
     threshold_method: max_val_loss
     use_dst_node_loss: True
diff --git a/pidsmaker/config/config.py b/pidsmaker/config/config.py
index dcb3175..60fb597 100644
--- a/pidsmaker/config/config.py
+++ b/pidsmaker/config/config.py
@@ -1,6 +1,46 @@
 import os

 DATASET_DEFAULT_CONFIG = {
+    "THEIA_E3_CUT": {
+        "raw_dir": "",
+        "database": "theia_e3_cut",
+        "database_all_file": "theia_e3_cut",
+        "num_node_types": 3,
+        "num_edge_types": 10,
+        "timezone": "US/Eastern",
+        "start_date": "2018-04-02",
+        "end_date": "2018-04-14",
+        "train_dates": [
+            "2018-04-02",
+            "2018-04-03",
+            "2018-04-04",
+            "2018-04-05",
+            "2018-04-06",
+            "2018-04-07",
+            "2018-04-08",
+            "2018-04-10"
+        ],
+        "val_dates": [
+            "2018-04-09"
+        ],
+        "test_dates": [
+            "2018-04-12",
+            "2018-04-13"
+        ],
+        "unused_dates": [
+            "2018-04-11"
+        ],
+        "ground_truth_relative_path": [
+            "E3-THEIA/node_test_after_cutoff.csv"
+        ],
+        "attack_to_time_window": [
+            [
+                "E3-THEIA/node_test_after_cutoff.csv",
+                "2018-04-12 12:35:00",
+                "2018-04-14 00:00:00"
+            ]
+        ]
+    },
     "THEIA_E5": {
         "raw_dir": "",
         "database": "theia_e5",
@@ -1180,6 +1220,14 @@ TASK_ARGS = {
             bool,
             desc="Whether the featurization method should be trained on all datasets in `multi_dataset`.",
         ),
+        "multi_dataset_primary_test_only": Arg(
+            bool,
+            desc="Cross-host mode: the main dataset given on the CLI is used ONLY for testing. Its train/val "
+            "splits are excluded from featurization and GNN training (thresholds come from the other "
+            "datasets' validation splits), and TGN state (memory, neighbor loader) is reset between "
+            "datasets because node ids of different datasets overlap. Requires `construction.multi_dataset` "
+            "and both `multi_dataset_training` flags.",
+        ),
         "pretrain_datasets": Arg(
             str,
             desc="Comma-separated list of dataset names for multi-dataset pretraining (used by word2vec, doc2vec, etc.).",
@@ -1427,6 +1475,12 @@ TASK_ARGS = {
                 bool, desc="Whether to cluster nodes after thresholding as done in Orthrus"
             ),
             "kmeans_top_K": Arg(int, desc="Number of top-score nodes selected before clustering."),
+            "only_subject_nodes": Arg(
+                bool,
+                desc="Evaluate only subject (process) nodes: file and netflow nodes are removed "
+                "from both the scored nodes and the ground truth before computing metrics. "
+                "The detector, its threshold and its training are unchanged.",
+            ),
             "ocrapt_contamination": Arg(
                 float,
                 desc="For threshold_method=ocrapt: max per-type contamination (top fraction "
diff --git a/pidsmaker/config/pipeline.py b/pidsmaker/config/pipeline.py
index 190da2e..e60046e 100644
--- a/pidsmaker/config/pipeline.py
+++ b/pidsmaker/config/pipeline.py
@@ -32,6 +32,9 @@ ROOT_GROUND_TRUTH_DIR = os.path.join(ROOT_PROJECT_PATH, "Ground_Truth/")
 # ================================================================================


+_WARNED_ONCE = set()  # warnings printed once per process (validation runs once per dataset/task)
+
+
 def get_default_cfg(args):
     """
     Inits the shared cfg object with default configurations.
@@ -677,6 +680,31 @@ def check_edge_cases(cfg):
                 "Using multi-dataset mode requires setting `preprocessing.construction.multi_dataset`"
             )

+    if getattr(cfg.featurization, "multi_dataset_primary_test_only", False):
+        if not (
+            use_multi_dataset
+            and cfg.featurization.multi_dataset_training
+            and cfg.batching.multi_dataset_training
+        ):
+            raise ValueError(
+                "`featurization.multi_dataset_primary_test_only` requires `construction.multi_dataset` "
+                "and both `featurization.multi_dataset_training` and `batching.multi_dataset_training` to be True"
+            )
+    elif (
+        use_multi_dataset
+        and (cfg.featurization.multi_dataset_training or cfg.batching.multi_dataset_training)
+        and "primary_in_training" not in _WARNED_ONCE
+    ):
+        _WARNED_ONCE.add("primary_in_training")
+        print(
+            "[WARNING] Multi-dataset training is ON but `featurization.multi_dataset_primary_test_only` is OFF: "
+            f"the train/val splits of the main dataset ({getattr(cfg.dataset, 'name', '?')}) ARE used for training. "
+            "For cross-host evaluation (train on the other datasets, test on the main one) add "
+            "--featurization.multi_dataset_primary_test_only=True",
+            file=sys.stderr,
+            flush=True,
+        )
+
     if cfg.evaluation.used_method == "edge_evaluation":
         if cfg._is_node_level:
             raise ValueError("Edge evaluation not implemented for node-level detection.")
diff --git a/pidsmaker/detection/evaluation_methods/node_evaluation.py b/pidsmaker/detection/evaluation_methods/node_evaluation.py
index 23803f5..418166e 100644
--- a/pidsmaker/detection/evaluation_methods/node_evaluation.py
+++ b/pidsmaker/detection/evaluation_methods/node_evaluation.py
@@ -37,6 +37,10 @@ from pidsmaker.detection.evaluation_methods.evaluation_utils import (
     reduce_losses_to_score,
     transform_attack2nodes_to_node2attacks,
 )
+from pidsmaker.detection.evaluation_methods.node_type_filter import (
+    filter_attack_gps_by_type,
+    filter_results_by_type,
+)
 from pidsmaker.utils.labelling import get_GP_of_each_attack
 from pidsmaker.detection.evaluation_methods.plot import plot_scores_distribution
 from pidsmaker.utils.utils import (
@@ -334,6 +338,16 @@ def main(

     results, thr, edge_records = get_preds_fn(cfg=cfg, val_tw_path=val_tw_path, test_tw_path=test_tw_path)

+    node_to_path = get_node_to_path_and_type(cfg)
+
+    # Optional: evaluate on subject (process) nodes only. Filters the scored nodes and the
+    # ground truth below; the detector, its threshold and its training are unchanged.
+    only_subject = bool(getattr(cfg.evaluation.node_evaluation, "only_subject_nodes", False))
+    if only_subject:
+        n_scored = len(results)
+        results = filter_results_by_type(results, node_to_path, "subject")
+        log(f"only_subject_nodes=True: evaluating {len(results)}/{n_scored} scored nodes (subject only)")
+
     # save results for future checking
     # os.makedirs(cfg.evaluation._results_dir, exist_ok=True)
     # results_save_dir = os.path.join(cfg.evaluation._results_dir, "results.pth")
@@ -342,8 +356,6 @@ def main(
     # torch.save(results, results_epoch_save_dir)
     # log(f"Resutls saved to {results_save_dir}")

-    node_to_path = get_node_to_path_and_type(cfg)
-
     out_dir = cfg.evaluation._precision_recall_dir
     os.makedirs(out_dir, exist_ok=True)

@@ -365,6 +377,10 @@ def main(
     distrib_img_file = os.path.join(out_dir, f"distrib_{model_epoch_dir}.png")

     attack_to_GPs = get_GP_of_each_attack(cfg)
+    if only_subject:
+        attack_to_GPs = filter_attack_gps_by_type(attack_to_GPs, node_to_path, "subject")
+        log("only_subject_nodes=True: ground truth per attack (subject nodes): "
+            + ", ".join(f"attack {k}: {len(v['nids'])}" for k, v in attack_to_GPs.items()))
     attack2nodes = {k: v["nids"] for k, v in attack_to_GPs.items()}
     node2attacks = transform_attack2nodes_to_node2attacks(attack2nodes)
     attack_to_TPs = defaultdict(int)
diff --git a/pidsmaker/detection/training_methods/inference_loop.py b/pidsmaker/detection/training_methods/inference_loop.py
index 9593e84..981035f 100644
--- a/pidsmaker/detection/training_methods/inference_loop.py
+++ b/pidsmaker/detection/training_methods/inference_loop.py
@@ -279,6 +279,8 @@ def main(cfg, model, val_data, test_data, epoch, split, logging=True, train_data
     model.to_device(device)

     model_epoch_file = f"model_epoch_{epoch}"
+    # Cross-host mode: node ids of different datasets overlap, so TGN memory must be cleared per dataset
+    reset_between_datasets = bool(getattr(cfg.featurization, "multi_dataset_primary_test_only", False))
     if use_cuda:
         torch.cuda.reset_peak_memory_stats(device=device)

@@ -295,6 +297,8 @@ def main(cfg, model, val_data, test_data, epoch, split, logging=True, train_data

         all_losses = []
         for graphs in dataset:
+            if reset_between_datasets and hasattr(model, "reset_state"):
+                model.reset_state()
             for g in log_tqdm(graphs, desc=desc, logging=logging):
                 g.to(device=device)

diff --git a/pidsmaker/detection/training_methods/training_loop.py b/pidsmaker/detection/training_methods/training_loop.py
index a00401c..318b2ed 100644
--- a/pidsmaker/detection/training_methods/training_loop.py
+++ b/pidsmaker/detection/training_methods/training_loop.py
@@ -100,6 +100,8 @@ def main(cfg):
     global_best_val_score = float("-inf")
     use_few_shot = cfg.training.decoder.use_few_shot
     grad_acc = cfg.training.grad_accumulation
+    # Cross-host mode: node ids of different datasets overlap, so TGN memory must not carry over
+    reset_between_datasets = bool(getattr(cfg.featurization, "multi_dataset_primary_test_only", False))

     es_tracker = early_stop_tracker_factory(cfg)
     # Stable optim: warmup cosine scheduler (computed over total training steps)
@@ -130,6 +132,8 @@ def main(cfg):
             loss_acc = torch.zeros(1, device=device)
             tot_loss = 0
             for dataset in train_data:
+                if reset_between_datasets:
+                    model.reset_state()
                 for i, g in enumerate(log_tqdm(dataset, "Training")):
                     g.to(device=device)
                     g = remove_attacks_if_needed(g, cfg)
diff --git a/pidsmaker/utils/data_utils.py b/pidsmaker/utils/data_utils.py
index 1276bf4..23595a7 100644
--- a/pidsmaker/utils/data_utils.py
+++ b/pidsmaker/utils/data_utils.py
@@ -32,7 +32,12 @@ from pidsmaker.utils.dataset_utils import (
     get_rel2id,
     get_possible_events,
 )
-from pidsmaker.utils.utils import get_multi_datasets, log_dataset_stats, log_tqdm
+from pidsmaker.utils.utils import (
+    get_multi_datasets,
+    get_training_datasets,
+    log_dataset_stats,
+    log_tqdm,
+)
 from pidsmaker.hetero import compute_hetero_features


@@ -176,7 +181,10 @@ def load_data_set(cfg, split: str, multi_dataset=False) -> list[CollatableTempor
     Returns a list of time window graphs for a given `split` (train/val/test set).
     """
     if multi_dataset:
-        multi_datasets = get_multi_datasets(cfg)
+        # Cross-host mode: the main dataset is test-only, so it is dropped from train/val.
+        multi_datasets = (
+            get_training_datasets(cfg) if split in ("train", "val") else get_multi_datasets(cfg)
+        )
         all_data_lists = []
         for dataset in multi_datasets:
             updated_cfg, _ = update_cfg_for_multi_dataset(cfg, dataset)
@@ -619,6 +627,9 @@ def run_intra_graph_batching(datasets, full_data, device, max_node, cfg, graph_r
                 tgn_loader_cfg=tgn_loader_cfg,
                 node_feat_dim=sample.x_src.shape[1],
                 node_type_dim=sample.node_type_src.shape[1],
+                reset_between_datasets=bool(
+                    getattr(cfg.featurization, "multi_dataset_primary_test_only", False)
+                ),
             )

         else:
@@ -636,6 +647,7 @@ def compute_tgn_graphs(
     tgn_loader_cfg,
     node_feat_dim,
     node_type_dim,
+    reset_between_datasets=False,
 ):
     tgn_neighbor_n_hop = tgn_loader_cfg.tgn_neighbor_n_hop
     fix_tgn_neighbor_loader = tgn_loader_cfg.fix_tgn_neighbor_loader
@@ -652,8 +664,19 @@ def compute_tgn_graphs(
     node_type_cache = torch.zeros((max_node, node_type_dim), device=device)
     assoc = torch.empty(max_node, dtype=torch.long, device=device)

+    # Neighbor-loader edge ids restart at 0 after `reset_state()`, but `full_data` is the concatenation
+    # of ALL datasets, so ids must be shifted by the number of edges inserted before the reset.
+    edge_offset = 0
+
     for dataset in datasets:
         for data_list in dataset:
+            if reset_between_datasets:
+                # Node ids of different datasets overlap (each DB indexes nodes from 0): without a reset,
+                # neighbors of a node in one dataset would leak into the same id in another dataset.
+                edge_offset += neighbor_loader.cur_e_id
+                neighbor_loader.reset_state()
+                node_feat_cache.zero_()
+                node_type_cache.zero_()
             for batch in log_tqdm(data_list, desc="Computing TGN last neighbor graphs"):
                 batch = batch.to(device)
                 batch_edge_index = batch.edge_index.clone()
@@ -670,8 +693,8 @@ def compute_tgn_graphs(
                     # NOTE: TGN's loader wrongly index edges (less than 1% in the returned e_id and edge_index)
                     # https://github.com/pyg-team/pytorch_geometric/issues/10100
                     # Should be replaced by an actual fix when available
-                    real_src = full_data.src[e_id.cpu()].to(device)
-                    real_dst = full_data.dst[e_id.cpu()].to(device)
+                    real_src = full_data.src[e_id.cpu() + edge_offset].to(device)
+                    real_dst = full_data.dst[e_id.cpu() + edge_offset].to(device)

                     loader_src = n_id[edge_index[0]]
                     loader_dst = n_id[edge_index[1]]
@@ -722,10 +745,10 @@ def compute_tgn_graphs(
                 batch.n_id_tgn = n_id
                 batch.edge_index_tgn = edge_index
                 batch.reindexed_edge_index_tgn = assoc[batch.edge_index]
-                batch.msg_tgn = full_data.msg[e_id.cpu()]
-                batch.t_tgn = full_data.t[e_id.cpu()]
+                batch.msg_tgn = full_data.msg[e_id.cpu() + edge_offset]
+                batch.t_tgn = full_data.t[e_id.cpu() + edge_offset]
                 batch.node_type_tgn = node_type_cache[n_id]
-                batch.edge_type_tgn = full_data.edge_type[e_id.cpu()]
+                batch.edge_type_tgn = full_data.edge_type[e_id.cpu() + edge_offset]

                 batch = batch.to("cpu")

diff --git a/pidsmaker/utils/dataset_utils.py b/pidsmaker/utils/dataset_utils.py
index be7e202..9784b21 100644
--- a/pidsmaker/utils/dataset_utils.py
+++ b/pidsmaker/utils/dataset_utils.py
@@ -467,6 +467,7 @@ ntype2id = {
 }

 DARPA_TC_DATASETS = {
+    "THEIA_E3_CUT",
     "CADETS_E3",
     "CADETS_E5",
     "THEIA_E5",
diff --git a/pidsmaker/utils/utils.py b/pidsmaker/utils/utils.py
index 5642fcb..c345b63 100644
--- a/pidsmaker/utils/utils.py
+++ b/pidsmaker/utils/utils.py
@@ -501,7 +501,7 @@ def get_indexid2msg(cfg, gather_multi_dataset=False):
         return indexid2msg

     if gather_multi_dataset:
-        multi_datasets = get_multi_datasets(cfg)
+        multi_datasets = get_training_datasets(cfg)
         log(f"Multi-dataset order: {multi_datasets}")

         all_indexid2msg = {}
@@ -527,7 +527,7 @@ def get_split2nodes(cfg, gather_multi_dataset=False):
         split2nodes = torch.load(path)
         return split2nodes

-    multi_datasets = get_multi_datasets(cfg)
+    multi_datasets = get_training_datasets(cfg)
     use_multi_dataset = "none" not in cfg.construction.multi_dataset

     if gather_multi_dataset and use_multi_dataset:
@@ -701,3 +701,15 @@ def get_multi_datasets(cfg):
         map(lambda x: x.strip(), multi_dataset.split("," if "," in multi_dataset else "-"))
     )
     return [cfg.dataset.name] + [d for d in multi_datasets if d != cfg.dataset.name]
+
+
+def get_training_datasets(cfg):
+    """
+    Datasets whose train/val splits feed featurization and GNN training.
+    Same as `get_multi_datasets`, except in cross-host mode
+    (`featurization.multi_dataset_primary_test_only`) where the main dataset is test-only.
+    """
+    datasets = get_multi_datasets(cfg)
+    if getattr(cfg.featurization, "multi_dataset_primary_test_only", False):
+        datasets = datasets[1:]
+    return datasets
__PIDSM_PATCH__
if git apply --check "$PATCH_TMP" >/dev/null 2>&1; then git apply "$PATCH_TMP"; echo "[patch] da ap patch loi PIDSMaker"; else echo "[patch] patch da co san hoac xung dot -> bo qua"; fi
rm -f "$PATCH_TMP"

# Chay tat ca: tu nap DB thieu, tao index, 4 bai, metric, don dia, verify, notify
chmod +x run_all.sh run_parallel.sh verify_final.sh vm_setup.sh download_datasets.sh
bash ./run_all.sh
````

Muốn đổi hành vi thì sửa các dòng `export ...` ở đầu ô (ví dụ `SEEDS=1`, `PARALLEL=1`, `RESTORE=force`, `ONLY=A_velox`, `SKIP_SETUP=1`) rồi chạy lại.
