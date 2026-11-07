"""Real local UI scenarios. Study uses honestly labeled local text, not fake Wikipedia."""
import os,json,sys,tempfile,threading,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from werkzeug.serving import make_server
from playwright.sync_api import sync_playwright
from crystal_lab import create_app
from imitation_learning import TrainingExample


def main():
    root_repo=Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory)
        subprocess.run([sys.executable,str(root_repo/'examples/lab_demo.py'),str(root)],check=True,capture_output=True)
        app=create_app(root)
        # Only preexisting split metadata is registered; the user never authors JSON.
        app.extensions['lab'].examples=[TrainingExample(**e) for e in json.loads((root/'lab-examples.json').read_text(encoding='utf8'))]
        server=make_server('127.0.0.1',0,app,threaded=True);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        base=f'http://127.0.0.1:{server.server_port}'
        try:
            with sync_playwright() as p:
                options=dict(headless=True,args=['--no-sandbox','--enable-unsafe-swiftshader'])
                if os.environ.get('CRYSTAL_LAB_BROWSER'):options['executable_path']=os.environ['CRYSTAL_LAB_BROWSER']
                browser=p.chromium.launch(**options);page=browser.new_page(viewport={'width':1500,'height':1100})
                errors=[];external=[];page.on('pageerror',lambda e:errors.append(str(e)))
                def route(r):
                    if not r.request.url.startswith(base+'/'):external.append(r.request.url);r.abort()
                    else:r.continue_()
                page.route('**/*',route);page.goto(base)
                page.wait_for_function("() => document.querySelector('#nodes').textContent !== '—'")
                # A: actual core processing of a clearly labeled local observation source.
                page.click('[data-tab="study"]');page.fill('#study-topic','яблоко')
                page.locator('#study details').first.locator('summary').click()
                page.fill('#study-material','Яблоко это фрукт. Яблоко вызывает рост. Фрукт это пища.')
                with page.expect_response(lambda r:r.url.endswith('/api/study/start')) as response:page.click('#study-start')
                job_id=response.value.json()['id']
                page.wait_for_function("() => document.querySelector('#study-status').textContent.includes('изменения применены')")
                assert 'local_text' in page.locator('#study-source').inner_text()
                assert 'Такт мышления' in page.locator('#study-events').inner_text()
                assert 'Сон' in page.locator('#study-events').inner_text()
                job=app.extensions['jobs'].entries[job_id];job.thread.join(5)
                assert job.done and job.state=='completed'
                assert len(app.extensions['lab'].population.active.resonators)>1
                assert not any(r.metadata.get('status')=='confirmed' for key,r in app.extensions['lab'].population.active.edge_meta.items() if len(key)==3)
                destination=Path(os.environ.get('CRYSTAL_LAB_SCREENSHOTS',directory));destination.mkdir(exist_ok=True,parents=True)
                page.screenshot(path=str(destination/'crystal-lab-study.png'),full_page=True)
                (destination/'study-scenario.json').write_text(json.dumps(job.events,ensure_ascii=False,indent=2),encoding='utf8')
                # B: separately load established, confirmed fixture knowledge.
                page.click('[data-tab="memory"]');page.select_option('#snapshot','lab-demo.cry')
                with page.expect_response(lambda r:r.url.endswith('/api/snapshots/load')) as response:page.click('#snapshot-load')
                instance=response.value.json()['instance']
                page.wait_for_function("id => document.querySelector('#nodes').textContent === '11' && document.querySelector('#instance').value === String(id)",arg=instance)
                assert app.extensions['lab'].population.active.semantic_space is not None
                page.click('[data-tab="imitation"]')
                assert not page.locator('#expert-mode').evaluate('(e) => e.open')
                page.fill('#simple-question','Что такое яблоко?');page.fill('#simple-reference','Яблоко является фруктом.')
                with page.expect_response(lambda r:r.url.endswith('/api/imitation/start')) as response:page.click('#simple-start')
                job_id=response.value.json()['id']
                page.wait_for_function("() => document.querySelector('#imitation-status').textContent.includes('изменения применены')")
                assert 'Исходная попытка' in page.locator('#imitation-events').inner_text()
                assert 'Проверен кандидат' in page.locator('#imitation-events').inner_text()
                job=app.extensions['jobs'].entries[job_id];job.thread.join(5)
                assert any(e['stage']=='candidate' for e in job.events)
                assert app.extensions['lab'].population.active.imitation_state['parameters']['is_a_form']==1
                page.screenshot(path=str(destination/'crystal-lab-imitation.png'),full_page=True)
                (destination/'imitation-scenario.json').write_text(json.dumps(job.events,ensure_ascii=False,indent=2),encoding='utf8')
                # Unsupported semantics: honest refusal, no replacement, same form kept for retry.
                page.fill('#simple-reference','Яблоко растёт на яблоне.');page.click('#simple-start')
                page.wait_for_function("() => document.querySelector('#imitation-status').textContent.includes('рабочее ядро не заменено')")
                assert page.locator('#imitation-help').is_visible()
                page.click('#offer-study');assert page.input_value('#study-topic')=='яблоко'
                page.click('#retry-imitation');assert page.input_value('#simple-reference')=='Яблоко растёт на яблоне.'
                assert not errors,errors;assert not external,external
                browser.close()
                print('PASS A: local-source study, observations, actual ticks/sleep, diff, no automatic confirmation')
                print('PASS B: question/reference only, real initial/candidate events, guarded calibration, honest refusal, retry link')
        finally:server.shutdown();thread.join(5)

if __name__=='__main__':main()
