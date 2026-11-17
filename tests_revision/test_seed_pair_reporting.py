import csv,json,tempfile,unittest
from pathlib import Path
from report_seed_pair_results import summarize,FIELDS

class ReportingTests(unittest.TestCase):
    def records(self):
        rows=[]
        for seed in [11,12]:
            for pair in ['a','b']:
                for episode in [0,1]:
                    success=int(seed==11 or episode==0)
                    rows.append(dict(record_kind='measured',episode_id=f'{seed}-{pair}-{episode}',benchmark='B',group='G',seed=seed,pair_id=pair,checkpoint_steps=400000,success=success,spl=.5*success))
        return rows
    def run_rows(self,rows):
        with tempfile.TemporaryDirectory()as tmp:
            p=Path(tmp)/'episodes.csv'
            with p.open('w',newline='')as f:
                w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader();w.writerows(rows)
            return summarize(p,Path(tmp)/'out')
    def test_balanced_statistics(self):
        r=self.run_rows(self.records());s=r['summary'][0]
        self.assertEqual(s['SR_mean_percent'],75)
        self.assertAlmostEqual(s['SR_sample_SD_percent'],35.35533905932738)
        self.assertEqual(s['SPL_mean'],.375)
        self.assertEqual(len(r['per_pair_seed']),4)
    def test_duplicate_rejected(self):
        rows=self.records();rows.append(rows[0].copy())
        with self.assertRaisesRegex(ValueError,'Duplicate'):self.run_rows(rows)
    def test_invalid_spl_rejected(self):
        rows=self.records();rows[-1]['spl']=1
        with self.assertRaisesRegex(ValueError,'SPL'):self.run_rows(rows)
    def test_mixed_provenance_rejected(self):
        rows=self.records();rows[-1]['record_kind']='synthetic_reference'
        with self.assertRaisesRegex(ValueError,'record_kind'):self.run_rows(rows)
    def test_synthetic_provenance_preserved(self):
        rows=self.records()
        for r in rows:r['record_kind']='synthetic_reference'
        r=self.run_rows(rows)
        self.assertEqual(r['manifest']['record_kind'],'synthetic_reference')
        self.assertTrue(all(x['record_kind']=='synthetic_reference'for x in r['per_pair']))
    def test_unmatched_pair_counts_rejected(self):
        rows=self.records()[:-1]
        with self.assertRaisesRegex(ValueError,'counts'):self.run_rows(rows)

if __name__=='__main__':unittest.main()
