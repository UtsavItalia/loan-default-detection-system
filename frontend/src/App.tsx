import { useState, useEffect } from 'react';
import './chartSetup';
import { getPrediction, getModelStats, getFeatureImportance } from './api';
import type {
  PredictionRequest,
  PredictionResponse,
  ModelStatsResponse,
  FeatureImportanceResponse,
} from './api';

import { Bar } from 'react-chartjs-2';

const defaultForm: PredictionRequest = {
  EXT_SOURCE_1: 0.5,
  EXT_SOURCE_2: 0.5,
  EXT_SOURCE_3: 0.5,
  DAYS_BIRTH: -10000,
  DAYS_EMPLOYED: -2000,
  DAYS_REGISTRATION: -5000,
  DAYS_ID_PUBLISH: -3000,
  DAYS_LAST_PHONE_CHANGE: -500,
  REGION_RATING_CLIENT: 2,
  REGION_RATING_CLIENT_W_CITY: 2,
  REG_CITY_NOT_WORK_CITY: 0,
  FLAG_EMP_PHONE: 1,
  FLAG_DOCUMENT_3: 1,
  AMT_CREDIT: 500000,
  AMT_GOODS_PRICE: 450000,
};

export default function App() {
  const [form, setForm] = useState<PredictionRequest>(defaultForm);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [modelStats, setModelStats] = useState<ModelStatsResponse | null>(null);
  const [featureImportance, setFeatureImportance] = useState<FeatureImportanceResponse | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    async function loadDashboardData() {
      try {
        const [stats, importance] = await Promise.all([
          getModelStats(),
          getFeatureImportance(),
        ]);
        setModelStats(stats);
        setFeatureImportance(importance);
      } catch (err) {
        console.error('Failed to load dashboard data:', err);
      }
    }
    loadDashboardData();
  }, []);

  const handlePredict = async () => {
    setLoading(true);
    try {
      const result = await getPrediction(form);
      setPrediction(result);
    } catch (err) {
      console.error('Prediction failed:', err);
    } finally {
      setLoading(false);
    }
  };

  const updateField = (field: keyof PredictionRequest, value: string) => {
    setForm(prev => ({ ...prev, [field]: parseFloat(value) || 0 }));
  };
  return (
    <div className="min-h-screen bg-dark-950 p-6">
      {/* Header */}
      <header className="mb-8">
        <div className="flex items-center gap-3 mb-2">
          <div className="w-2 h-8 bg-accent-cyan rounded-full" />
          <h1 className="text-3xl font-bold tracking-tight font-[family-name:var(--font-display)]">
            Loan Default Detection
          </h1>
        </div>
        <p className="text-sm text-slate-400 ml-5">
          ML-powered credit risk assessment with SHAP explainability
        </p>
      </header>
      {/* Main Grid */}
      <div className="grid grid-cols-12 gap-5">

        {/* Left Column — Prediction Form */}
        <div className="col-span-4 space-y-5">

          <div className="bg-dark-800 border border-glass-border rounded-2xl p-5 card-glow">
            <h2 className="text-sm font-semibold text-accent-cyan uppercase tracking-widest mb-4 font-[family-name:var(--font-mono)]">
              Applicant Details
            </h2>

            <div className="space-y-3">
              {Object.entries(form).map(([key, value]) => (
                <div key={key}>
                  <label className="block text-xs text-slate-400 mb-1 font-[family-name:var(--font-mono)]">
                    {key}
                  </label>
                  <input
                    type="number"
                    value={value}
                    onChange={(e) => updateField(key as keyof PredictionRequest, e.target.value)}
                    className="w-full bg-dark-900 border border-glass-border rounded-lg px-3 py-2 text-sm text-slate-200 font-[family-name:var(--font-mono)] focus:outline-none focus:border-accent-cyan transition-colors"
                  />
                </div>
              ))}
            </div>

            <button
              onClick={handlePredict}
              disabled={loading}
              className="w-full mt-5 py-3 bg-accent-cyan text-dark-950 font-semibold rounded-xl hover:brightness-110 transition-all disabled:opacity-50 cursor-pointer"
            >
              {loading ? 'Analyzing...' : 'Run Prediction'}
            </button>
          </div>
        </div>
        {/* Right Column — Results & Charts */}
        <div className="col-span-8 space-y-5">

          {/* Risk Result Card */}
          {prediction && (
            <div className={`border rounded-2xl p-5 ${prediction.risk_level === 'Low'
              ? 'bg-emerald-500/5 border-emerald-500/20'
              : prediction.risk_level === 'Medium'
                ? 'bg-amber-500/5 border-amber-500/20'
                : 'bg-rose-500/5 border-rose-500/20'
              }`}>
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-xs text-slate-400 uppercase tracking-widest font-[family-name:var(--font-mono)] mb-1">Risk Assessment</p>
                  <p className={`text-4xl font-bold ${prediction.risk_level === 'Low'
                    ? 'text-accent-emerald'
                    : prediction.risk_level === 'Medium'
                      ? 'text-accent-amber'
                      : 'text-accent-rose'
                    }`}>
                    {prediction.risk_level} Risk
                  </p>
                </div>
                <div className="text-right">
                  <p className="text-xs text-slate-400 font-[family-name:var(--font-mono)]">Default Probability</p>
                  <p className="text-3xl font-bold text-slate-100 font-[family-name:var(--font-mono)]">
                    {(prediction.probability * 100).toFixed(1)}%
                  </p>
                  <p className="text-xs text-slate-400 font-[family-name:var(--font-mono)] mt-1">
                    Prediction: {prediction.prediction === 1 ? 'DEFAULT' : 'NO DEFAULT'}
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Model Metrics Cards */}
          {modelStats && (
            <div className="grid grid-cols-4 gap-3">
              {Object.entries(modelStats.model_results)
                .sort(([, a], [, b]) => b.roc_auc - a.roc_auc)
                .map(([name, metrics], index) => {
                  const isBest = index === 0;
                  return (
                    <div
                      key={name}
                      className={`relative rounded-2xl p-4 border ${isBest
                          ? 'bg-accent-cyan/5 border-accent-cyan/20'
                          : 'bg-dark-800 border-glass-border'
                        }`}
                    >
                      {isBest && (
                        <span className="absolute -top-2 right-3 bg-accent-cyan text-dark-950 text-[10px] font-bold px-2 py-0.5 rounded-full font-[family-name:var(--font-mono)]">
                          BEST
                        </span>
                      )}
                      <p className="text-xs text-slate-400 font-[family-name:var(--font-mono)] mb-3">
                        {name}
                      </p>

                      <div className="space-y-2">
                        <div className="flex justify-between items-center">
                          <span className="text-[10px] text-slate-500 uppercase tracking-wider">ROC-AUC</span>
                          <span className="text-sm font-bold text-accent-cyan font-[family-name:var(--font-mono)]">
                            {metrics.roc_auc.toFixed(4)}
                          </span>
                        </div>

                        <div className="flex justify-between items-center">
                          <span className="text-[10px] text-slate-500 uppercase tracking-wider">Recall</span>
                          <span className="text-sm font-semibold text-accent-emerald font-[family-name:var(--font-mono)]">
                            {metrics.recall.toFixed(4)}
                          </span>
                        </div>

                        <div className="flex justify-between items-center">
                          <span className="text-[10px] text-slate-500 uppercase tracking-wider">Precision</span>
                          <span className="text-sm font-semibold text-accent-violet font-[family-name:var(--font-mono)]">
                            {metrics.precision.toFixed(4)}
                          </span>
                        </div>

                        <div className="flex justify-between items-center">
                          <span className="text-[10px] text-slate-500 uppercase tracking-wider">F1 Score</span>
                          <span className="text-sm font-semibold text-accent-amber font-[family-name:var(--font-mono)]">
                            {metrics.f1_score.toFixed(4)}
                          </span>
                        </div>
                      </div>

                      {/* Mini ROC-AUC bar */}
                      <div className="mt-3 h-1 bg-dark-900 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-accent-cyan rounded-full transition-all duration-500"
                          style={{ width: `${metrics.roc_auc * 100}%` }}
                        />
                      </div>
                    </div>
                  );
                })}
            </div>
          )}

          {/* Charts Row */}
          <div className="grid grid-cols-2 gap-5">

            {/* Model Comparison Chart */}
            <div className="bg-dark-800 border border-glass-border rounded-2xl p-5 card-glow">
              <h2 className="text-sm font-semibold text-accent-cyan uppercase tracking-widest mb-4 font-[family-name:var(--font-mono)]">
                Model Comparison — ROC-AUC
              </h2>
              <div className="h-64">
                {modelStats ? (
                  <Bar
                    data={{
                      labels: Object.keys(modelStats.model_results),
                      datasets: [
                        {
                          label: 'ROC-AUC',
                          data: Object.values(modelStats.model_results).map(m => m.roc_auc),
                          backgroundColor: ['#22d3ee', '#a78bfa', '#34d399', '#fbbf24'],
                          borderRadius: 6,
                        },
                      ],
                    }}
                    options={{
                      responsive: true,
                      maintainAspectRatio: false,
                      plugins: {
                        legend: { display: false },
                        tooltip: {
                          callbacks: {
                            label: (ctx) => `ROC-AUC: ${ctx.parsed.y?.toFixed(4) ?? 'N/A'}`,
                          },
                        },
                      },
                      scales: {
                        x: {
                          ticks: { color: '#94a3b8', font: { family: 'JetBrains Mono', size: 10 } },
                          grid: { display: false },
                        },
                        y: {
                          min: 0.5,
                          max: 0.8,
                          ticks: { color: '#64748b', font: { family: 'JetBrains Mono', size: 10 } },
                          grid: { color: 'rgba(255,255,255,0.03)' },
                        },
                      },
                    }}
                  />
                ) : (
                  <p className="text-slate-500 text-sm">Loading...</p>
                )}
              </div>
            </div>

            {/* Feature Importance Chart */}
            <div className="bg-dark-800 border border-glass-border rounded-2xl p-5 card-glow">
              <h2 className="text-sm font-semibold text-accent-cyan uppercase tracking-widest mb-4 font-[family-name:var(--font-mono)]">
                Feature Importance — SHAP
              </h2>
              <div className="h-64">
                {featureImportance ? (
                  <Bar
                    data={{
                      labels: Object.keys(featureImportance.feature_importance).map(
                        (name) => name.replace(/_/g, ' ')
                      ),
                      datasets: [
                        {
                          label: 'SHAP Value',
                          data: Object.values(featureImportance.feature_importance),
                          backgroundColor: '#a78bfa',
                          borderRadius: 4,
                        },
                      ],
                    }}
                    options={{
                      indexAxis: 'y',
                      responsive: true,
                      maintainAspectRatio: false,
                      plugins: {
                        legend: { display: false },
                        tooltip: {
                          callbacks: {
                            label: (ctx) => `SHAP: ${ctx.parsed.x?.toFixed(6) ?? 'N/A'}`,
                          },
                        },
                      },
                      scales: {
                        x: {
                          ticks: { color: '#64748b', font: { family: 'JetBrains Mono', size: 9 } },
                          grid: { color: 'rgba(255,255,255,0.03)' },
                        },
                        y: {
                          ticks: { color: '#94a3b8', font: { family: 'JetBrains Mono', size: 9 } },
                          grid: { display: false },
                        },
                      },
                    }}
                  />
                ) : (
                  <p className="text-slate-500 text-sm">Loading...</p>
                )}
              </div>
            </div>

            {/* Confusion Matrix */}
            <div className="bg-dark-800 border border-glass-border rounded-2xl p-5 card-glow col-span-2">
              <h2 className="text-sm font-semibold text-accent-cyan uppercase tracking-widest mb-4 font-[family-name:var(--font-mono)]">
                Confusion Matrix — Best Model
              </h2>
              <div className="h-48 flex items-center justify-center">
                {modelStats ? (() => {
                  const bestName = Object.entries(modelStats.model_results)
                    .sort(([, a], [, b]) => b.roc_auc - a.roc_auc)[0];
                  const cm = bestName[1].confusion_matrix;

                  return (
                    <div className="w-full max-w-md">
                      <p className="text-xs text-slate-400 text-center mb-3 font-[family-name:var(--font-mono)]">
                        {bestName[0]}
                      </p>
                      <div className="grid grid-cols-[auto_1fr_1fr] grid-rows-[auto_1fr_1fr] gap-1 text-center">
                        {/* Header row */}
                        <div />
                        <p className="text-xs text-slate-400 font-[family-name:var(--font-mono)] pb-1">Pred: 0</p>
                        <p className="text-xs text-slate-400 font-[family-name:var(--font-mono)] pb-1">Pred: 1</p>

                        {/* Row 1 — Actual 0 */}
                        <p className="text-xs text-slate-400 font-[family-name:var(--font-mono)] pr-3 flex items-center">Actual: 0</p>
                        <div className="bg-emerald-500/10 border border-emerald-500/20 rounded-xl p-3">
                          <p className="text-lg font-bold text-accent-emerald font-[family-name:var(--font-mono)]">
                            {cm[0][0].toLocaleString()}
                          </p>
                          <p className="text-[10px] text-slate-400">True Neg</p>
                        </div>
                        <div className="bg-rose-500/10 border border-rose-500/20 rounded-xl p-3">
                          <p className="text-lg font-bold text-accent-rose font-[family-name:var(--font-mono)]">
                            {cm[0][1].toLocaleString()}
                          </p>
                          <p className="text-[10px] text-slate-400">False Pos</p>
                        </div>

                        {/* Row 2 — Actual 1 */}
                        <p className="text-xs text-slate-400 font-[family-name:var(--font-mono)] pr-3 flex items-center">Actual: 1</p>
                        <div className="bg-rose-500/10 border border-rose-500/20 rounded-xl p-3">
                          <p className="text-lg font-bold text-accent-rose font-[family-name:var(--font-mono)]">
                            {cm[1][0].toLocaleString()}
                          </p>
                          <p className="text-[10px] text-slate-400">False Neg</p>
                        </div>
                        <div className="bg-emerald-500/10 border border-emerald-500/20 rounded-xl p-3">
                          <p className="text-lg font-bold text-accent-emerald font-[family-name:var(--font-mono)]">
                            {cm[1][1].toLocaleString()}
                          </p>
                          <p className="text-[10px] text-slate-400">True Pos</p>
                        </div>
                      </div>
                    </div>
                  );
                })() : (
                  <p className="text-slate-500 text-sm">Loading...</p>
                )}
              </div>
            </div>

          </div>
        </div>
      </div>
    </div>
  );
}

