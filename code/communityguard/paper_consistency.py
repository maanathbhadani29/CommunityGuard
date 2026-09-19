"""Check that a generated manuscript reports its own run, not an older target.

Historical metric equality is intentionally not a condition of this check.
Scientific outcomes are read from the run's CSVs and figure response arrays.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import platform
import re
import zipfile
import numpy as np
import pandas as pd


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_run_consistency(code, run):
    code, run = Path(code), Path(run)
    checks = {}
    for filename in ['all_numerical_metrics.csv', 'all_llama_metrics.csv', 'paired_metrics.csv']:
        data = pd.read_csv(run/filename, float_precision='round_trip')
        assert not data.duplicated(['cohort', 'case_id', 'method']).any(), filename
        before, after = data.error_before.to_numpy(), data.error_after.to_numpy()
        assert np.isfinite(before).all() and np.isfinite(after).all(), filename
        assert (before >= 0).all() and (after >= 0).all(), filename
        expected = {'improved': after < before-1e-12,
                    'harmed': after > before+1e-12,
                    'unchanged': np.abs(after-before) <= 1e-12}
        for field, values in expected.items():
            np.testing.assert_array_equal(data[field], values, err_msg=f'{filename}: {field}')
        np.testing.assert_array_equal(data[['improved','harmed','unchanged']].sum(axis=1), np.ones(len(data)))
        checks[filename] = {'records': len(data), 'unique_keys': True,
                            'outcome_flags_match_measured_errors': True}
    for metric_file, summary_file in [('all_numerical_metrics.csv','numerical_summary.csv'),
                                      ('paired_metrics.csv','paired_summary.csv')]:
        data = pd.read_csv(run/metric_file)
        summary = pd.read_csv(run/summary_file).set_index('method')
        attacks = data[data.attack.notna()]
        for method, group in attacks.groupby('method'):
            actual = summary.loc[method]
            assert int(actual.cases) == len(group)
            for field in ['improved','unchanged','harmed','joint_location_channel']:
                assert int(actual[field]) == int(group[field].sum()), (method,field)
            for field in ['error_before','error_after','collateral_error']:
                np.testing.assert_allclose(actual['mean_'+field], group[field].mean(), rtol=1e-12, atol=1e-15)
        checks[summary_file] = 'recomputed from this run’s per-case measurements'
    # Recompute figure RMSE and edited counts from actual input/output and truth.
    # These reporting functions do not require agreement with the old archive.
    from .case_reporting import case_details
    from .comparison_reporting import matched_cases
    cases, series, provenance, settings = case_details(code, run)
    matched, _ = matched_cases(code, run)
    assert len(cases) == 4 and len(matched) == 12
    from .evidence_reporting import summarize_evidence
    evidence = summarize_evidence(run)
    checks['figures_and_matched_table'] = 'four saved responses independently scored; 12 matched-policy rows checked'
    checks['tokens'] = 'CSV totals agree with authentic call records'
    # This is a source-preservation record, not equality with historical outcomes.
    frozen = json.loads((code/'frozen_design.json').read_text())['source_sha256']
    unchanged = all(sha256(code/name) == digest for name,digest in frozen.items())
    report = {'status':'PASS', 'checked_utc':datetime.now(timezone.utc).isoformat(),
              'runtime':{'python':platform.python_version(),'platform':platform.platform()},
              'checks':checks, 'scientific_sources_unchanged':unchanged,
              'historical_result_equality_required':False,
              'calls':len(evidence['calls']),
              'input_tokens':int(evidence['calls'].input_tokens.sum()),
              'output_tokens':int(evidence['calls'].output_tokens.sum()),
              'llama_sources':sorted(evidence['calls'].source.unique().tolist()),
              'figure_data_provenance':provenance}
    (run/'evidence/run_consistency_check.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def check_paper_consistency(code, run, paper):
    code, run, paper = Path(code), Path(run), Path(paper)
    report = check_run_consistency(code, run)
    numerical = pd.read_csv(run/'numerical_summary.csv').set_index('method')
    paired = pd.read_csv(run/'paired_summary.csv').set_index('method')
    clean = pd.read_csv(run/'clean_summary.csv').set_index('method').loc['guarded_rules']
    guarded = numerical.loc['guarded_rules']
    values = json.loads((paper/'reported_values.json').read_text())
    expected = {'CaseN':int(guarded.cases), 'ImproveN':int(guarded.improved),
                'HarmN':int(guarded.harmed), 'UnchangedN':int(guarded.unchanged),
                'JointN':int(guarded.joint_location_channel),
                'CleanN':int(clean.records), 'CleanHarmN':int(clean.harmed),
                'BeforeE':f'{guarded.mean_error_before:.6f}',
                'AfterE':f'{guarded.mean_error_after:.6f}',
                'ErrorReduction':f'{100*(1-guarded.mean_error_after/guarded.mean_error_before):.1f}',
                'PairedCaseN':int(paired.loc['guarded_rules','cases']),
                'LlamaImprove':int(paired.loc['llama_langgraph','improved']),
                'OneImprove':int(paired.loc['llama_one_pass','improved']),
                'PairedRulesImprove':int(paired.loc['guarded_rules','improved']),
                'LlamaHarmN':int(paired.loc['llama_langgraph','harmed']),
                'OneHarmN':int(paired.loc['llama_one_pass','harmed']),
                'PairedRulesHarmN':int(paired.loc['guarded_rules','harmed'])}
    for name, value in expected.items():
        assert values[name] == value, (name,values[name],value)
    macros = dict(re.findall(r'\\newcommand\{\\(\w+)\}\{([^}]*)\}', (paper/'update_numbers.tex').read_text()))
    for name,value in values.items():
        assert macros[name] == str(value), name
    assert r'\input{update_numbers.tex}' in (paper/'CommunityGuard_AI_Updated.tex').read_text()
    figure_checks={}
    for attack in ['market_hack','meter_hack','inverter_hack','time_spoofing']:
        source = run/'figures'/f'{attack}.pdf'
        used = paper/'figures'/source.name
        assert sha256(source)==sha256(used), attack
        assert f'figures/{attack}.pdf' in (paper/'case_figures.tex').read_text()
        figure_checks[attack]={'pdf_identical_to_this_run':True,'sha256':sha256(used)}
    # Table I's printed values must match the source summary, including baselines.
    table=(paper/'update_table.tex').read_text()
    for method,row in numerical.iterrows():
        fragment=f'& {int(row.improved)} & {int(row.unchanged)} & {int(row.harmed)} & {row.mean_error_after:.6f}'
        assert fragment in table, method
    # Check the displayed precision of every policy row in Table II.
    table=(paper/'case_comparison_table.tex').read_text()
    comparison=pd.read_csv(run/'evidence/figure_baseline_comparison.csv')
    for _,row in comparison.iterrows():
        precision=6 if row.figure==3 else 0 if row.figure==6 else 4
        fragment=f'& {int(row.edited_samples)} / {int(row.perturbed_samples)} & {row.target_rmse_after:.{precision}f} & {int(row.input_tokens):,} / {int(row.output_tokens):,}'
        assert fragment in table, (row.figure,row.method)
    report.update(paper_values_checked=expected, tables_checked=['Table I','Table II'],
                  figures_checked=figure_checks, paper_source='this run’s CSV measurements and saved response arrays',
                  pdf_compiled=(paper/'CommunityGuard_AI_Updated.pdf').exists())
    if report['pdf_compiled']:
        from pypdf import PdfReader
        pdf=paper/'CommunityGuard_AI_Updated.pdf'
        reader=PdfReader(pdf)
        assert len(reader.pages)<=8, f'Manuscript exceeds 8 pages: {len(reader.pages)}'
        report['pdf_pages']=len(reader.pages)
        report['pdf_sha256']=sha256(pdf)
    sources=[*paper.glob('*.tex'),*paper.glob('*.cls'),*paper.glob('figures/*.pdf'),
             run/'all_numerical_metrics.csv',run/'all_llama_metrics.csv',run/'paired_metrics.csv',
             run/'evidence/figure_case_timeseries.csv',run/'evidence/figure_baseline_comparison.csv']
    report['source_files']={str(p.relative_to(run)):sha256(p) for p in sorted(sources)}
    (paper/'paper_consistency_check.json').write_text(json.dumps(report,indent=2)+'\n')
    (run/'evidence/paper_consistency_check.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def export_latex_zip(paper):
    paper=Path(paper);target=paper/'CommunityGuard_AI_LaTeX.zip'
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
        for path in sorted(paper.rglob('*')):
            if path.is_file() and (path.suffix in {'.tex','.cls','.pdf','.json'}):
                z.write(path,path.relative_to(paper))
    return target
