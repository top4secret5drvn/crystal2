"""Live UI proof of observed answers, explicit evidence and conditional calibration."""
import os,sys,tempfile,threading,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from werkzeug.serving import make_server
from playwright.sync_api import sync_playwright
from crystal_lab import create_app


def main():
    with tempfile.TemporaryDirectory() as directory:
        app=create_app(directory);lab=app.extensions['lab']
        server=make_server('127.0.0.1',0,app,threaded=True);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        base=f'http://127.0.0.1:{server.server_port}'
        try:
            with sync_playwright() as p:
                options=dict(headless=True,args=['--no-sandbox','--enable-unsafe-swiftshader'])
                if os.environ.get('CRYSTAL_LAB_BROWSER'):options['executable_path']=os.environ['CRYSTAL_LAB_BROWSER']
                browser=p.chromium.launch(**options);page=browser.new_page(viewport={'width':1500,'height':1100});errors=[];external=[]
                page.on('pageerror',lambda e:errors.append(str(e)))
                def route(r):
                    if r.request.url.startswith(base+'/'):r.continue_()
                    else:external.append(r.request.url);r.abort()
                page.route('**/*',route);page.goto(base)
                page.wait_for_function("() => document.querySelector('#nodes').textContent !== '—'")
                for command in ('ввод яблоко это фрукт','спроси яблоко','ответь'):
                    page.fill('#command',command);page.click('#command-form button')
                    page.wait_for_function("() => !document.querySelector('#command-form button').disabled")
                assert 'Attributed' in page.locator('#console-output').inner_text()
                page.fill('#answer-question','Что такое яблоко?');page.click('#answer-form button')
                page.wait_for_function("() => document.querySelector('#answer-evidence').textContent.includes('Attributed')")
                page.select_option('#answer-mode','verified');page.click('#answer-form button')
                page.wait_for_function("() => document.querySelector('#answer-evidence').textContent.includes('Unknown')")
                page.click('[data-tab="study"]');page.fill('#study-topic','яблоко');page.locator('#study details').first.locator('summary').click()
                page.fill('#study-material','Яблоко — это фрукт. Яблоко является плодом яблони. Апельсин — цитрусовый плод. Яблоко происходит от яблони.')
                with page.expect_response(lambda r:r.url.endswith('/api/study/start')) as response:page.click('#study-start')
                study=app.extensions['jobs'].entries[response.value.json()['id']]
                page.wait_for_function("() => document.querySelector('#study-status').textContent.includes('изменения применены')")
                study.thread.join(5)
                report=lab.session.membrane.observation_report()
                assert report['definitions']>=3 and any(r['unsupported_relation']=='ORIGIN' for r in report['unparsed'])
                page.click('[data-tab="imitation"]')
                def set_pair(subject,reference):
                    page.fill('#simple-question','Что такое '+subject+'?');page.fill('#simple-reference',reference)
                def attempt():
                    with page.expect_response(lambda r:r.url.endswith('/api/imitation/start')) as response:page.click('#simple-start')
                    job=app.extensions['jobs'].entries[response.value.json()['id']]
                    page.wait_for_function("() => !document.querySelector('#simple-start').disabled")
                    job.thread.join(5);assert job.done;return job
                set_pair('яблоко','Яблоко — это плод яблони.');missing=attempt();assert not missing.events[-1]['committed']
                with page.expect_response(lambda r:r.url.endswith('/api/imitation/observation')) as response:page.click('#add-observation')
                witness=response.value.json();assert witness['status']=='user_observed' and not witness['model_parameters_changed']
                assert not witness['calibration_ready']
                set_pair('апельсин','Апельсин — это цитрусовый плод.');assert not attempt().events[-1]['committed']
                with page.expect_response(lambda r:r.url.endswith('/api/imitation/observation')) as response:page.click('#add-observation')
                assert response.value.json()['confirmed'] is False
                set_pair('яблоко','Яблоко — это плод яблони.');trained=attempt();assert trained.events[-1]['committed']
                result=next(e for e in trained.events if e['stage']=='round_complete')
                assert result['evidence_mode']=='user_observed';assert result['outcome']['new_evidence']==0
                assert result['outcome']['control_changes'];assert result['result']['changed']['is_a_construction']==[0,1]
                assert lab.session.membrane.respond('яблоко','verified')['mode']=='Unknown'
                assert lab.session.membrane.respond('яблоко')['mode']=='Attributed'
                page.fill('#answer-question','Что такое яблоко?');page.select_option('#answer-mode','auto');page.click('#answer-form button')
                page.wait_for_function("() => document.querySelector('#answer-evidence').textContent.includes('user_observed')")
                destination=Path(os.environ.get('CRYSTAL_LAB_SCREENSHOTS',directory));destination.mkdir(exist_ok=True,parents=True)
                page.screenshot(path=str(destination/'observation-answer-learning.png'),full_page=True)
                (destination/'observation-learning-trace.json').write_text(json.dumps(dict(report=report,missing=missing.events,witness=witness,training=trained.events),ensure_ascii=False,indent=2),encoding='utf8')
                assert not errors,errors;assert not external,external
                browser.close();print('PASS: clean text → Attributed; Verified stays blocked; study reports parsed/unparsed; explicit witness and independent conditional controls → real calibration')
        finally:server.shutdown();thread.join(5)

if __name__=='__main__':main()
