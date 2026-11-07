"""Live loopback UI: observed ambiguity -> corrections -> accepted mutation."""
import os,sys,tempfile,threading
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from crystal_lab import create_app
from werkzeug.serving import make_server
from playwright.sync_api import sync_playwright

def main():
 with tempfile.TemporaryDirectory() as directory:
  app=create_app(directory);server=make_server('127.0.0.1',0,app,threaded=True);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
  try:
   with sync_playwright() as p:
    options={'headless':True,'args':['--no-sandbox']}
    if os.environ.get('CRYSTAL_LAB_BROWSER'):options['executable_path']=os.environ['CRYSTAL_LAB_BROWSER']
    browser=p.chromium.launch(**options);page=browser.new_page();errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto(f'http://127.0.0.1:{server.server_port}');page.wait_for_function("()=>document.querySelector('#nodes').textContent!=='—'")
    for text in ('Двигатель состоит из деталей','Прибор состоит из модулей'):
     page.fill('#command','ввод '+text);page.click('#command-form button');page.wait_for_function("()=>!document.querySelector('#command-form button').disabled")
    page.click('[data-tab="semantic"]')
    page.get_by_role('button',name='деталей PART_OF двигатель',exact=True).click()
    page.wait_for_function("()=>document.querySelector('#semantic-result').textContent.includes('missing_independent_controls')")
    page.get_by_role('button',name='модулей PART_OF прибор',exact=True).click()
    page.wait_for_function("()=>document.querySelector('#semantic-result').textContent.includes('train_gain_without_control_regression')")
    assert not errors,errors;browser.close();print('PASS: real questions, two explicit corrections, guarded rule mutation')
  finally:server.shutdown();thread.join(5)
if __name__=='__main__':main()
