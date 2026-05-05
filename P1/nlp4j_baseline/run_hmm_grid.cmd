@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "TEST_PATH=..\datasets\auto\test.txt"
set "MODEL_PATH=model\hmm_bmes_model.json"
set "OUTPUT_ROOT=output_grid"

if not exist "%TEST_PATH%" (
  echo Missing test: %TEST_PATH%
  exit /b 1
)
if not exist "%MODEL_PATH%" (
  echo Missing model: %MODEL_PATH%
  exit /b 1
)

if not exist "%OUTPUT_ROOT%" mkdir "%OUTPUT_ROOT%"

for %%I in (-0.2 -0.5 -1.0) do (
  for %%S in (-0.2 -0.6 -1.0) do (
    set ITP=%%I
    set SP=%%S
    set EP=%%S

    set TAG=itp%%I_sp%%S_ep%%S
    set TAG=!TAG:-=m!
    set TAG=!TAG:.=p!

    set "OUTDIR=!OUTPUT_ROOT!\!TAG!"
    if not exist "!OUTDIR!" mkdir "!OUTDIR!"
    if not exist "!OUTDIR!\" (
      echo Failed to create output dir: !OUTDIR!
      exit /b 1
    )

    echo Running itp=%%I sp=%%S ep=%%S
    mvn -q exec:java -Dexec.mainClass=Nlp4jHmmPredictor -Dexec.args="%TEST_PATH% %MODEL_PATH% output %%I %%S %%S"
    if errorlevel 1 exit /b 1

    if not exist "output\nlp4j_hmm_metrics.json" (
      echo Missing output\nlp4j_hmm_metrics.json
      exit /b 1
    )
    if not exist "output\nlp4j_hmm_label_report.tsv" (
      echo Missing output\nlp4j_hmm_label_report.tsv
      exit /b 1
    )
    if not exist "output\nlp4j_hmm_confusion_matrix.tsv" (
      echo Missing output\nlp4j_hmm_confusion_matrix.tsv
      exit /b 1
    )

    copy /Y output\nlp4j_hmm_metrics.json "!OUTDIR!\" >nul
    copy /Y output\nlp4j_hmm_label_report.tsv "!OUTDIR!\" >nul
    copy /Y output\nlp4j_hmm_confusion_matrix.tsv "!OUTDIR!\" >nul
  )
)

echo Done. See %OUTPUT_ROOT%
endlocal