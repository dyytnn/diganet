# Publishing this repository as a Code Ocean compute capsule

The capsule reproduces **every figure and table of both manuscripts** from the released result
files. It needs no GPU and no image data, so a reviewer can press *Reproducible Run* and get the
published numbers back without obtaining the dataset first.

What the capsule does **not** do is train: the 249 Hoffman-modulation-contrast images of Saeedi et
al. belong to their original authors and are not ours to redistribute, and the time-lapse frames are
patient data from a partner clinic. Training from scratch is documented in the top-level `README.md`.

## One-time setup

1. **Create the capsule.** On codeocean.com: *New Capsule → Create New*. Pick a Python 3.11 starter
   environment (a PyTorch CPU starter saves a long install).
2. **Set the environment.** Open the Environment editor and paste `requirements.txt` into the pip
   section. Code Ocean writes `environment/Dockerfile` for you; do not hand-write it.
3. **Get the git URL and a token.** Capsule menu → *Clone via Git* for the URL. Account page →
   *Generate New Access Token*; it is shown once, so save it. The git username is your Code Ocean
   login email and the password is that token — an institutional (SSO) login has no usable password.
4. **Push the code.**

   ```bash
   codeocean/sync_capsule.sh https://git.codeocean.com/capsule-xxxxxxx.git
   ```

   The script copies the committed tree into the capsule's `code/` folder, installs `run` as the
   entry point and pushes. Only committed files travel, so an untracked checkpoint cannot leak.
5. **Run it** and check the results pane: 11 figures and the analysis tables, plus
   `benchmark_diagnostic_TE.json`.

## Updating it later

Commit here as usual, then run `sync_capsule.sh` again with the same URL.

## Publishing

Keep the capsule private through peer review and share it with the editor by link. When the paper is
accepted, submit the capsule for publication: Code Ocean verifies that it runs, then issues a DOI and
a BibTeX citation for the Code Availability statement.

## Layout, for reference

A capsule repository is not a mirror of this one. Its root is

```
code/          this repository, with `run` as the entry point; the working directory at run time
data/          read-only inputs — unused here, every input is a result file under code/results/
environment/   Dockerfile, generated from the Environment editor
results/       everything the run writes; nothing written elsewhere is kept
metadata/      title, authors, licence
```
