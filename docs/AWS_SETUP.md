# AWS Setup: running this pipeline on an EC2 CPU box (us-east-1)

This is a runbook for moving `code/business_entity_resolution/` off a
laptop with limited RAM onto a bigger CPU box when local runs get too slow or
start hitting memory pressure. It assumes no AWS experience beyond having an
account. **Never put AWS access keys, secret keys, or `.pem` file contents in
code, commits, or this repo.** Keys live only in your local `~/.aws/`
credentials file or environment variables on your own machine, never checked in.

## 0. Before you start: set a budget alert

Do this first, before launching anything, so a forgotten instance can't run
up an unexpected bill.

1. AWS Console -> search "Billing" -> **Billing and Cost Management**.
2. Left nav -> **Budgets** -> **Create budget** -> "Customize (advanced)".
3. Budget type: **Cost budget**, period **Monthly**, set an amount (e.g. $20).
4. Add an alert threshold (e.g. 80% of budget) and enter your email.
5. Save. You'll get an email if spend approaches the limit.

## 1. Launch an EC2 instance

1. Console -> **EC2** -> make sure the region selector (top right) says
   **us-east-1** (N. Virginia).
2. **Launch instance**.
3. Name: `ber-pipeline` (or anything).
4. AMI: **Ubuntu Server 22.04 LTS** (free-tier eligible AMI listing).
5. Instance type: for 64 GB RAM, use **r6i.2xlarge** (8 vCPU / 64 GB, memory-
   optimized -- this workload is RAM-bound, not compute-bound) or
   **m6i.4xlarge** (16 vCPU / 64 GB, more CPU if you want faster
   multiprocessing normalization). Both are CPU-only, no GPU, matching "no
   GPU needed for this step."
6. Key pair: **Create new key pair**, type ED25519, download the `.pem` file.
   Keep it out of any git repo (e.g. `~/.ssh/ber-pipeline.pem`), and on
   Linux/Mac run `chmod 400 ~/.ssh/ber-pipeline.pem`.
7. Network settings: allow **SSH (port 22)** from **My IP** only (not
   0.0.0.0/0 -- don't expose SSH to the whole internet).
8. Storage: bump the root volume to at least **200 GiB** gp3 (the raw dataset
   is ~2.5 GB, but parquet intermediates under `data/` plus OS/venv easily
   need headroom; 200 GiB gives margin).
9. **Launch instance**. Wait for status checks to pass (~1-2 min).

## 2. Connect over SSH

```bash
ssh -i ~/.ssh/ber-pipeline.pem ubuntu@<instance-public-ip>
```

(Public IP is on the instance's detail page in the console. It changes on
stop/start unless you allocate an Elastic IP, which isn't necessary for a
throwaway compute box.)

## 3. Install Python + set up the venv

```bash
sudo apt-get update
sudo apt-get install -y python3.10 python3.10-venv git tmux
git clone <your-private-fork-or-remote-url> ber
cd ber/student_resource
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r code/business_entity_resolution/requirements.txt
```

If the repo isn't pushed anywhere yet, skip `git clone` and copy the code
directory directly with `scp` (step 4 covers the same transfer method for
the dataset).

## 4. Get the dataset onto the instance

The raw `dataset/` folder is gitignored and non-trivial in size (~2.5 GB
across 7 TSVs), so don't put it in git. Two options:

**Option A -- private S3 bucket (recommended for repeated runs):**

On your laptop (with your own AWS credentials configured via `aws configure`,
never on the instance):

```bash
aws s3 mb s3://<your-unique-bucket-name> --region us-east-1
aws s3 sync student_resource/dataset s3://<your-unique-bucket-name>/dataset
```

On the EC2 instance, attach an **IAM role** with S3 read access to the
instance (EC2 console -> instance -> Actions -> Security -> Modify IAM role)
instead of putting credentials on the box, then:

```bash
aws s3 sync s3://<your-unique-bucket-name>/dataset ~/ber/student_resource/dataset
```

**Option B -- direct scp (fine for a one-off run):**

```bash
scp -i ~/.ssh/ber-pipeline.pem -r student_resource/dataset ubuntu@<instance-ip>:~/ber/student_resource/
```

Either way, delete the bucket / the copy on the instance when you're done if
the data is sensitive or you don't want to keep paying for S3 storage.

## 5. Run the pipeline inside tmux

Always run long jobs inside `tmux` so they survive your SSH connection
dropping:

```bash
tmux new -s ber
cd ~/ber/student_resource
source .venv/bin/activate
python code/business_entity_resolution/src/ingest.py
python code/business_entity_resolution/src/mine_dicts.py
python code/business_entity_resolution/src/normalize.py
python code/business_entity_resolution/src/folds.py
python code/business_entity_resolution/src/evaluate.py
python code/business_entity_resolution/src/blocking.py
python code/business_entity_resolution/src/features.py
# ... model training script, once it exists
```

Detach with `Ctrl+b` then `d` (job keeps running). Reattach later with
`tmux attach -t ber`. Check progress any time by reattaching or by tailing a
redirected log file (`python ... > run.log 2>&1 &`, then `tail -f run.log`).

## 6. Get results back

```bash
# from your laptop
scp -i ~/.ssh/ber-pipeline.pem -r ubuntu@<instance-ip>:~/ber/student_resource/output ./output-from-ec2
```

## 7. STOP the instance when you're done for the day

**Stop, don't just disconnect** -- a stopped instance bills only for its EBS
storage, not compute:

```bash
# from your laptop, with your own AWS CLI configured
aws ec2 stop-instances --instance-ids <instance-id> --region us-east-1
```

or EC2 console -> select instance -> **Instance state** -> **Stop instance**.
When you're fully done with the box, **Terminate** it instead (deletes the
root volume too, so make sure `output/` and anything else you need is
already copied off per step 6).

## 8. Requesting a GPU quota increase (only if a later step needs one)

Nothing in this step requires a GPU (LightGBM here runs on CPU, and the
license constraint caps us at MIT/Apache-2.0, <=8B-parameter models with no
GPU-only requirement). If a later step wants a GPU instance (e.g. a
`g5.xlarge` for a heavier model) or a SageMaker GPU notebook, EC2 GPU
instances are gated behind a service quota that's 0 by default on new
accounts:

1. Console -> **Service Quotas** -> **AWS services** -> search "EC2".
2. Find the quota matching the instance family, e.g. **"Running On-Demand G
   and VT instances"** for `g5`/`g4dn`.
3. **Request increase**, enter the vCPU count you need (e.g. 4 for one
   `g5.xlarge`), submit.
4. For SageMaker specifically, quotas live under the **SageMaker** service
   in the same Service Quotas console (e.g. "ml.g5.xlarge for notebook
   instance usage").
5. Approval can take from minutes to ~1 business day for a small request.

## Cost awareness

`r6i.2xlarge` and `m6i.4xlarge` are on the order of $0.40-$0.90/hour
on-demand in us-east-1 (check current EC2 pricing before relying on this).
A few hours for the full pipeline (ingest -> mine_dicts -> normalize ->
blocking -> features -> train) should cost a few dollars. The budget alert
in step 0 and remembering to stop the instance (step 7) are the two things
that actually prevent a surprise bill.
