import os
from pathlib import Path
path = Path('---')

files = sorted([s for s in os.listdir(path) if s.endswith('.txt') and s.startswith('2026')])[::-1]


filename = files[0]
input_file = path / filename
output_file = 'html化/'+filename.replace('.txt','_1.html')

'''
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            border: 1px solid black;
            text-align: left;
            padding: 8px;
        }
        th {
            background-color: #f2f2f2;
        }
        .left-column {
            width: 40%;
        }
        .right-column {
            width: 60%;
        }
    </style>
</head>
<body>
    <script>
        document.addEventListener('DOMContentLoaded', (event) => {
            let undoStack = [];
            let redoStack = [];
            const table = document.querySelector('table');

            table.addEventListener('mouseup', (event) => {
                let selection = window.getSelection();
                if (!selection.isCollapsed) {
                    let range = selection.getRangeAt(0);
                    let span = document.createElement('span');
                    if (event.shiftKey) {
                        span.style.color = 'black';
                        span.style.fontWeight = 'normal';
                    } else {
                        span.style.color = 'red';
                        span.style.fontWeight = 'bold';
                    }
                    span.appendChild(range.extractContents());
                    range.insertNode(span);

                    // Save the current state to the undo stack
                    undoStack.push(table.innerHTML);
                    redoStack = []; // Clear the redo stack
                }
            });

            document.addEventListener('keydown', (event) => {
                if (event.ctrlKey && event.key === 'z') {
                    // UNDO
                    if (undoStack.length > 0) {
                        redoStack.push(table.innerHTML);
                        table.innerHTML = undoStack.pop();
                    }
                } else if (event.ctrlKey && event.key === 'y') {
                    // REDO
                    if (redoStack.length > 0) {
                        undoStack.push(table.innerHTML);
                        table.innerHTML = redoStack.pop();
                    }
                }
            });
        });
    </script>
</body>
</html>
'''

# Python code to generate the HTML file
def generate_html_table(input_file, output_file):
    # Read the input file
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split content based on '###' delimiter
    sections = content.split('###')
    sections_e = [s for s in sections[0].split('\n') if not s == ""]
    sections_j = [s for s in sections[1].split('\n') if not s == ""]

    #[print(s) for s in sections_e]
    #[print(s) for s in sections_j]
    
    # Prepare the HTML structure
    html_content = '''
    <!DOCTYPE html>
    <html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            border: 1px solid black;
            text-align: left;
            padding: 8px;
        }
        th {
            background-color: #f2f2f2;
        }
        .left-column {
            width: 60%;
        }
        .right-column {
            width: 40%;
        }
        .sticky-header {
            position: sticky;
            top: 0;
            background: white;
            z-index: 100;
        }
    </style>
</head>
        <body>
            <table class="sticky-header">
                <tr>
                <td contenteditable="true" colspan="2">&nbsp;</td>
                </tr>
            </table>
            <br>
            <table>
    '''

    # Loop through the sections to create table rows
    for i in range(max(len(sections_e),len(sections_j))):
        try:
            original_text = sections_e[i].strip()
        except:
            original_text = ""
        try:
            translated_text = sections_j[i].strip()
        except:
            translated_text = ""
                
        html_content += f'''
            <tr>
                <td class='left-column' contenteditable='true'>{original_text}</td>
                <td class='right-column' contenteditable='true'>{translated_text}</td>
            </tr>
            '''

    # Close the table and HTML tags
    html_content += '''
        </table>
<script>
document.addEventListener('DOMContentLoaded', () => {
  let undoStack = [];
  let redoStack = [];

  function saveState() {
    undoStack.push(document.body.innerHTML);
    redoStack = [];
  }

  function getEditableCellFromEventTarget(target) {
    return target?.closest?.('td[contenteditable="true"]') ?? null;
  }

  function selectionIsUsable(selection, cell) {
    if (!selection || selection.rangeCount === 0 || selection.isCollapsed) return false;
    const range = selection.getRangeAt(0);
    const common = range.commonAncestorContainer;
    const node = common.nodeType === Node.ELEMENT_NODE ? common : common.parentElement;
    return !!(node && cell && cell.contains(node));
  }

  // 選択範囲の中に「強調span」が1つでもあるか判定
  function selectionContainsHighlightSpan(range) {
    const HIGHLIGHT_SELECTOR = 'span[data-hl="1"]';
    const ancestor = range.commonAncestorContainer.nodeType === Node.ELEMENT_NODE
      ? range.commonAncestorContainer
      : range.commonAncestorContainer.parentElement;

    if (!ancestor) return false;

    // 1) ancestor配下の候補spanを走査し、rangeと交差するか
    const spans = ancestor.querySelectorAll(HIGHLIGHT_SELECTOR);
    for (const sp of spans) {
      if (range.intersectsNode(sp)) return true;
    }

    // 2) そもそも range が span の内側にあるケース（単一ノード選択等）
    let n = range.startContainer.nodeType === Node.ELEMENT_NODE
      ? range.startContainer
      : range.startContainer.parentElement;
    while (n) {
      if (n.matches && n.matches(HIGHLIGHT_SELECTOR)) return true;
      n = n.parentElement;
    }

    return false;
  }

  function clearFormattingInSelection(range) {
    const plainText = range.toString();
    range.deleteContents();
    range.insertNode(document.createTextNode(plainText));
  }

  function applyHighlightToSelection(range) {
    const span = document.createElement('span');
    // 判定を安定させるため data 属性で印を付ける
    span.setAttribute('data-hl', '1');
    span.style.color = 'red';
    span.style.fontWeight = 'bold';
    span.appendChild(range.extractContents());
    range.insertNode(span);
  }

  document.addEventListener('mouseup', (event) => {
    const cell = getEditableCellFromEventTarget(event.target);
    if (!cell) return;

    const selection = window.getSelection();
    if (!selectionIsUsable(selection, cell)) return;

    const range = selection.getRangeAt(0);

    // 変更前を保存
    saveState();

    const hasHighlight = selectionContainsHighlightSpan(range);
    if (hasHighlight) {
      // 強調が含まれるなら → 全解除
      clearFormattingInSelection(range);
    } else {
      // 強調がなければ → 全付与
      applyHighlightToSelection(range);
    }

    selection.removeAllRanges();
  }, { capture: true });

  // Undo/Redo（Mac: Cmd / Windows: Ctrl）
  document.addEventListener('keydown', (event) => {
    const isUndo = (event.key === 'z' || event.key === 'Z') && (event.metaKey || event.ctrlKey) && !event.shiftKey;
    const isRedo =
      ((event.key === 'y' || event.key === 'Y') && (event.metaKey || event.ctrlKey)) ||
      ((event.key === 'z' || event.key === 'Z') && (event.metaKey || event.ctrlKey) && event.shiftKey);

    if (isUndo) {
      event.preventDefault();
      if (undoStack.length > 0) {
        redoStack.push(document.body.innerHTML);
        document.body.innerHTML = undoStack.pop();
      }
    } else if (isRedo) {
      event.preventDefault();
      if (redoStack.length > 0) {
        undoStack.push(document.body.innerHTML);
        document.body.innerHTML = redoStack.pop();
      }
    }
  });
});
</script>
    </body>
        </html>
        '''

    # Write the HTML content to the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print('終わりました')