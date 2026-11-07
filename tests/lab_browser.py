"""Optional real-browser offline user journeys. Not part of API unit discovery.
Requires Playwright and Chromium. CRYSTAL_LAB_BROWSER selects a system executable.
"""
import hashlib,json,os,subprocess,sys,tempfile,threading
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from werkzeug.serving import make_server
from playwright.sync_api import sync_playwright
from crystal_lab import create_app


def main():
    repo=Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory)
        subprocess.run([sys.executable,str(repo/'examples/lab_demo.py'),str(root)],check=True,capture_output=True)
        source=(root/'lab-demo.cry').read_bytes()
        app=create_app(root);server=make_server('127.0.0.1',0,app,threaded=True)
        worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
        base=f'http://127.0.0.1:{server.server_port}'
        try:
            with sync_playwright() as p:
                options={'headless':True,'args':['--no-sandbox','--enable-unsafe-swiftshader']}
                if os.environ.get('CRYSTAL_LAB_BROWSER'):options['executable_path']=os.environ['CRYSTAL_LAB_BROWSER']
                browser=p.chromium.launch(**options);page=browser.new_page(viewport={'width':1500,'height':1100})
                errors=[];external=[]
                page.on('pageerror',lambda e:errors.append(str(e)))
                def route(r):
                    if not r.request.url.startswith(base+'/'):
                        external.append(r.request.url);r.abort()
                    else:r.continue_()
                page.route('**/*',route)
                page.goto(base);page.wait_for_function("() => document.querySelector('#nodes').textContent !== '—'")
                page.select_option('#snapshot','lab-demo.cry');page.click('#snapshot-load')
                page.wait_for_function("() => document.querySelector('#nodes').textContent === '11'")
                page.fill('#search','яблоко');page.click('#graph-load')
                page.wait_for_function("() => document.querySelector('#graph-info').textContent.startsWith('1 /')")
                box=page.locator('#graph').bounding_box();page.locator('#graph').click(position={'x':box['width']/2,'y':box['height']/2})
                page.wait_for_function("() => document.querySelector('#node-title').textContent === 'яблоко'")
                assert 'IS_A' in page.locator('#details').inner_text()
                page.click('#neighbors');page.click('#graph-reset')
                page.wait_for_function("() => document.querySelector('#graph-info').textContent.startsWith('10 /')")
                page.fill('#command','думай 1');page.click('#command-form button')
                page.wait_for_function("() => document.querySelector('#tick').textContent === '1'")
                page.click('[data-tab="history"]')
                page.wait_for_selector('#history-list details')
                page.locator('#history-list details').filter(has_text='думай 1').locator('summary').click()
                assert 'energy_changes' in page.locator('#history-list').inner_text()
                page.click('[data-tab="imitation"]');page.click('#expert-mode summary')
                page.fill('#examples-json',(root/'lab-examples.json').read_text(encoding='utf8'));page.click('#examples-import')
                page.wait_for_function("() => document.querySelector('#example').options.length === 1")
                assert 'IS_A' in page.locator('#gold').inner_text()
                for _ in range(2):
                    with page.expect_response(lambda r:r.url.endswith('/api/training/run')) as response:page.click('#train-run')
                    assert response.value.json()['accepted']
                    page.wait_for_function("() => document.querySelector('#training-report').textContent.includes('train_gain_without_control_regression')")
                with page.expect_response(lambda r:r.url.endswith('/api/training/rollback')) as response:page.click('#train-rollback')
                assert response.value.status==200
                page.click('[data-tab="geometry"]');page.select_option('#geometry-view','3');page.click('#geometry-load')
                page.wait_for_selector('#space canvas');assert '3D' in page.locator('#geometry-info').inner_text()
                if os.environ.get('CRYSTAL_LAB_SCREENSHOTS'):
                    destination=Path(os.environ['CRYSTAL_LAB_SCREENSHOTS']);destination.mkdir(parents=True,exist_ok=True)
                    page.screenshot(path=str(destination/'crystal-lab-geometry.png'),full_page=True)
                page.select_option('#geometry-view','8');page.click('#geometry-load')
                page.wait_for_function("() => document.querySelector('#space').textContent.includes('нет 8D')")
                page.click('[data-tab="memory"]');page.fill('#save-name','browser-save.cry')
                with page.expect_response(lambda r:r.url.endswith('/api/snapshots/save')) as response:page.click('#snapshot-save')
                assert response.value.status==200
                assert (root/'browser-save.cry').exists() and (root/'lab-demo.cry').read_bytes()==source
                page.click('[data-tab="population"]');page.select_option('#instance','0')
                page.wait_for_function("() => document.querySelector('#nodes').textContent === '1'")
                assert app.extensions['lab'].population.active_index==1
                page.select_option('#instance','1');page.click('[data-tab="memory"]')
                page.wait_for_function("() => document.querySelector('#nodes').textContent === '11'")
                if os.environ.get('CRYSTAL_LAB_SCREENSHOTS'):page.screenshot(path=str(destination/'crystal-lab.png'),full_page=True)
                assert not errors,errors
                assert not external,external
                browser.close()
                print('PASS: offline graph + inspector + command/diff + training/rollback + 3D + snapshot preservation + instance browsing; no external requests/page errors')
        finally:server.shutdown();worker.join(timeout=5)

if __name__=='__main__':main()
