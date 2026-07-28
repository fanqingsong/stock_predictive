import { useMemo } from 'react'
import { useParams } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import {
  Alert,
  Card,
  Col,
  Collapse,
  Descriptions,
  Progress,
  Row,
  Space,
  Spin,
  Statistic,
  Table,
  Tag,
  Typography,
} from 'antd'
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { api } from '../api'

function fmtPct(v) {
  return typeof v === 'number' ? `${v.toFixed(1)}%` : '-'
}

function fmtNum(v, digits = 4) {
  return typeof v === 'number' ? v.toFixed(digits) : v ?? '-'
}

function ScaleSeriesChart({ series, label }) {
  const data = useMemo(
    () => (series || []).map((p) => ({ date: p.date, close: p.close })),
    [series]
  )
  if (!data.length) {
    return <Typography.Text type="secondary">暂无该尺度 K 线</Typography.Text>
  }
  return (
    <div style={{ width: '100%', height: 220 }}>
      <ResponsiveContainer>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="date" hide />
          <YAxis domain={['auto', 'auto']} width={56} />
          <Tooltip />
          <Legend />
          <Line
            type="monotone"
            dataKey="close"
            name={`${label}收盘`}
            stroke="#1677ff"
            dot={false}
            strokeWidth={2}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

function HorizonCard({ item }) {
  const train = item.training?.train_metrics || {}
  const val = item.training?.val_metrics || {}
  const valAcc = val.direction_acc
  const missing = item.status && item.status !== 'ok'
  const up = item.direction === 'up'
  const arch = item.architecture || {}
  const training = item.training

  return (
    <Card
      title={
        <Space wrap>
          <span>{item.label}度模型</span>
          {missing ? (
            <Tag>{item.status === 'missing' ? '未训练' : '不可用'}</Tag>
          ) : (
            <Tag color={up ? 'success' : 'error'}>{up ? '看涨' : '看跌'}</Tag>
          )}
          {item.bar_rule ? <Tag>{item.bar_rule}</Tag> : null}
        </Space>
      }
    >
      <Typography.Paragraph type="secondary">{item.description}</Typography.Paragraph>

      {missing ? (
        <Alert type="warning" showIcon message={item.error || '该尺度模型不可用'} />
      ) : (
        <>
          <Statistic
            title="上涨概率"
            value={(item.probability ?? 0) * 100}
            precision={1}
            suffix="%"
          />
          <Progress
            percent={Math.round((item.probability ?? 0) * 100)}
            status={up ? 'success' : 'exception'}
            showInfo={false}
            style={{ marginTop: 8, marginBottom: 12 }}
          />
          <Row gutter={8}>
            <Col span={12}>
              <Statistic title="验证准确率" value={valAcc ?? '-'} suffix="%" />
            </Col>
            <Col span={12}>
              <Statistic title="训练准确率" value={train.direction_acc ?? '-'} suffix="%" />
            </Col>
          </Row>
          <Typography.Text type="secondary" style={{ display: 'block', marginTop: 8 }}>
            基准周期截止 {item.as_of}
            {item.ref_close != null ? ` · 收盘 ${Number(item.ref_close).toFixed(3)}` : ''}
          </Typography.Text>
        </>
      )}

      <Collapse
        size="small"
        style={{ marginTop: 16 }}
        items={[
          {
            key: 'arch',
            label: '网络结构',
            children: (
              <Space direction="vertical" style={{ width: '100%' }} size="middle">
                <Typography.Text code style={{ whiteSpace: 'pre-wrap' }}>
                  {arch.diagram || '-'}
                </Typography.Text>
                <Descriptions size="small" column={1} bordered>
                  <Descriptions.Item label="网络">{arch.network || 'StockLSTM'}</Descriptions.Item>
                  <Descriptions.Item label="输入形状">{arch.input_shape || '-'}</Descriptions.Item>
                  <Descriptions.Item label="lookback">{arch.lookback ?? '-'}</Descriptions.Item>
                  <Descriptions.Item label="特征维">{arch.input_size ?? '-'}</Descriptions.Item>
                  <Descriptions.Item label="LSTM 层数">{arch.num_layers ?? '-'}</Descriptions.Item>
                  <Descriptions.Item label="hidden">{arch.hidden_size ?? '-'}</Descriptions.Item>
                  <Descriptions.Item label="dropout">{arch.dropout ?? '-'}</Descriptions.Item>
                  <Descriptions.Item label="输出">{arch.head || 'logit → P(up)'}</Descriptions.Item>
                </Descriptions>
                <div>
                  <Typography.Text strong>特征列</Typography.Text>
                  <div style={{ marginTop: 8 }}>
                    {(arch.feature_columns || []).map((c) => (
                      <Tag key={c} style={{ marginBottom: 4 }}>
                        {c}
                      </Tag>
                    ))}
                    {!arch.feature_columns?.length ? '-' : null}
                  </div>
                </div>
              </Space>
            ),
          },
          {
            key: 'train',
            label: '训练数据与超参',
            children: training ? (
              <Descriptions size="small" column={1} bordered>
                <Descriptions.Item label="K 线规则">{training.bar_rule || item.bar_rule || '-'}</Descriptions.Item>
                <Descriptions.Item label="聚合后 K 线数">{training.n_bars ?? seriesLen(item)}</Descriptions.Item>
                <Descriptions.Item label="特征行数">{training.n_feature_rows ?? '-'}</Descriptions.Item>
                <Descriptions.Item label="训练/验证划分">
                  {training.train_ratio != null
                    ? `${Math.round(training.train_ratio * 100)}% / ${Math.round((1 - training.train_ratio) * 100)}%`
                    : '-'}
                  {training.split_idx != null ? `（split_idx=${training.split_idx}）` : ''}
                </Descriptions.Item>
                <Descriptions.Item label="训练序列数">{training.n_train_samples ?? '-'}</Descriptions.Item>
                <Descriptions.Item label="验证序列数">{training.n_val_samples ?? '-'}</Descriptions.Item>
                <Descriptions.Item label="epochs">
                  {training.epochs_ran ?? '-'}
                  {training.best_epoch != null ? `（最佳 ${training.best_epoch}）` : ''}
                  {training.patience != null ? ` / patience ${training.patience}` : ''}
                </Descriptions.Item>
                <Descriptions.Item label="batch / lr / wd">
                  {training.batch_size ?? '-'} / {fmtNum(training.lr, 4)} /{' '}
                  {fmtNum(training.weight_decay, 5)}
                </Descriptions.Item>
                <Descriptions.Item label="pos_weight">{fmtNum(training.pos_weight, 3)}</Descriptions.Item>
                <Descriptions.Item label="特征版本">{training.feature_version || item.feature_version || '-'}</Descriptions.Item>
                <Descriptions.Item label="训练时间">{training.trained_at || '-'}</Descriptions.Item>
                <Descriptions.Item label="验证指标">
                  acc {fmtPct(val.direction_acc)} · F1 {fmtPct(val.f1_up)} · n=
                  {val.n_samples ?? '-'}
                </Descriptions.Item>
                <Descriptions.Item label="训练指标">
                  acc {fmtPct(train.direction_acc)} · F1 {fmtPct(train.f1_up)} · n=
                  {train.n_samples ?? '-'}
                </Descriptions.Item>
              </Descriptions>
            ) : (
              <Typography.Text type="secondary">尚未训练，无训练记录</Typography.Text>
            ),
          },
          {
            key: 'series',
            label: `${item.label}尺度训练用 K 线（收盘）`,
            children: <ScaleSeriesChart series={item.series} label={item.label} />,
          },
        ]}
      />
    </Card>
  )
}

function seriesLen(item) {
  return item.series?.length ?? '-'
}

function ModelSummaryTable({ horizons }) {
  const rows = (horizons || []).map((h) => {
    const a = h.architecture || {}
    const t = h.training || {}
    return {
      key: h.key,
      label: h.label,
      status: h.status,
      input_shape: a.input_shape,
      layers: a.num_layers,
      hidden: a.hidden_size,
      dropout: a.dropout,
      n_bars: t.n_bars,
      n_feat: t.n_feature_rows,
      n_train: t.n_train_samples,
      n_val: t.n_val_samples,
      val_acc: t.val_metrics?.direction_acc,
    }
  })

  return (
    <Table
      size="small"
      pagination={false}
      dataSource={rows}
      columns={[
        { title: '尺度', dataIndex: 'label', width: 64 },
        {
          title: '状态',
          dataIndex: 'status',
          width: 80,
          render: (s) =>
            s === 'ok' ? <Tag color="success">就绪</Tag> : <Tag>{s || '-'}</Tag>,
        },
        { title: '输入形状', dataIndex: 'input_shape' },
        { title: 'LSTM 层', dataIndex: 'layers', width: 72 },
        { title: 'hidden', dataIndex: 'hidden', width: 72 },
        { title: 'dropout', dataIndex: 'dropout', width: 80 },
        { title: 'K 线数', dataIndex: 'n_bars', width: 72 },
        { title: '特征行', dataIndex: 'n_feat', width: 72 },
        { title: '训练样本', dataIndex: 'n_train', width: 88 },
        { title: '验证样本', dataIndex: 'n_val', width: 88 },
        {
          title: '验证准确率',
          dataIndex: 'val_acc',
          width: 100,
          render: (v) => fmtPct(v),
        },
      ]}
      scroll={{ x: 900 }}
    />
  )
}

export default function ResultPage() {
  const { ticker } = useParams()
  const decoded = decodeURIComponent(ticker)

  const { data, isLoading, error } = useQuery({
    queryKey: ['predict', decoded],
    queryFn: () => api.predict(decoded),
  })

  if (isLoading) return <Spin tip="加载预测结果..." />
  if (error) return <Alert type="error" message={error.message} showIcon />
  if (!data) return null

  const quote = data.quote || {}
  const horizons = data.horizons || []

  return (
    <div>
      <Space align="center" style={{ marginBottom: 16 }} wrap>
        <Typography.Title level={3} style={{ margin: 0 }}>
          {data.ticker} · {data.market_label}
        </Typography.Title>
        <Tag color="blue">日 / 自然周 / 自然月</Tag>
      </Space>

      <Alert
        type="info"
        showIcon
        style={{ marginBottom: 16 }}
        message="三套独立模型：各自使用对应尺度的 K 线与网络配置。下方可展开查看网络结构、训练样本与该尺度收盘序列。"
      />

      <Typography.Title level={5}>多尺度模型对照</Typography.Title>
      <ModelSummaryTable horizons={horizons} />

      <Row gutter={[16, 16]} style={{ marginTop: 24, marginBottom: 8 }}>
        {horizons.map((h) => (
          <Col xs={24} lg={8} key={h.key}>
            <HorizonCard item={h} />
          </Col>
        ))}
      </Row>

      <Typography.Paragraph type="secondary" style={{ marginTop: 16 }}>
        训练时间：{data.trained_at} · 特征版本：{data.feature_version} · 币种：
        {data.currency_label}
      </Typography.Paragraph>

      <Typography.Title level={5}>行情信息</Typography.Title>
      <Descriptions bordered size="small" column={{ xs: 1, sm: 2, md: 3 }}>
        <Descriptions.Item label="名称">{quote.Name || data.ticker}</Descriptions.Item>
        <Descriptions.Item label="现价">{quote.Last_Sale ?? '-'}</Descriptions.Item>
        <Descriptions.Item label="涨跌">{quote.Net_Change ?? '-'}</Descriptions.Item>
        <Descriptions.Item label="涨跌幅">{quote.Percent_Change ?? '-'}</Descriptions.Item>
        <Descriptions.Item label="成交量">{quote.Volume ?? '-'}</Descriptions.Item>
        <Descriptions.Item label="行业">{quote.Industry ?? data.market_label}</Descriptions.Item>
      </Descriptions>
    </div>
  )
}
