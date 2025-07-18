#!/bin/bash

# 現在のスクリプトのディレクトリを取得
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")

DEFAULT_FILE="aichallenge_submit.tar.gz"
TARGET_FILE="${1:-$DEFAULT_FILE}"

# 展開元のファイルパスを定義
SOURCE_FILE="${SCRIPT_DIR}/aichallenge/adapt_submit/${TARGET_FILE}"

# 展開先のディレクトリパスを定義
DEST_DIR="${SCRIPT_DIR}/aichallenge/workspace/src/"

# --- 実行前チェック ---
# 展開元のファイルが存在するか確認します。
if [ ! -f "${SOURCE_FILE}" ]; then
  echo "エラー: 対象ファイルが見つかりません。"
  echo "確認したパス: ${SOURCE_FILE}"
  exit 1
fi

# 展開先ディレクトリが存在しない場合は作成
# mkdir -p は、親ディレクトリが存在しない場合もまとめて作成し、
# 既に存在していてもエラーにしない便利なオプションです。
mkdir -p "${DEST_DIR}"

# tarコマンドでファイルを上書き展開
# -x: 展開 (extract)
# -z: gzip形式の圧縮を解凍 (gzip)
# -v: 処理中のファイル名を表示 (verbose)
# -f: 対象のアーカイブファイルを指定 (file)
# -C: 指定したディレクトリに移動してから展開 (Change directory)
echo "展開を開始します..."
echo "From: ${SOURCE_FILE}"
echo "To:   ${DEST_DIR}"

rm -rf "${DEST_DIR}"/aichallenge_submit

tar -xzvf "${SOURCE_FILE}" -C "${DEST_DIR}"

echo "展開が完了しました。"