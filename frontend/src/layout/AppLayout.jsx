import { Layout, Menu, theme, Typography } from 'antd'
import {
  HomeOutlined,
  LineChartOutlined,
  UnorderedListOutlined,
} from '@ant-design/icons'
import { Link, Outlet, useLocation } from 'react-router-dom'

const { Header, Sider, Content } = Layout

const items = [
  { key: '/', icon: <HomeOutlined />, label: <Link to="/">首页</Link> },
  {
    key: '/predict',
    icon: <LineChartOutlined />,
    label: <Link to="/predict">训练与预测</Link>,
  },
  {
    key: '/tickers',
    icon: <UnorderedListOutlined />,
    label: <Link to="/tickers">Ticker 列表</Link>,
  },
]

export default function AppLayout() {
  const location = useLocation()
  const {
    token: { colorBgContainer, borderRadiusLG },
  } = theme.useToken()

  const selected =
    items.find((i) => i.key !== '/' && location.pathname.startsWith(i.key))?.key ||
    (location.pathname === '/' ? '/' : '')

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Sider breakpoint="lg" collapsedWidth={64} width={220}>
        <div className="brand">
          <Typography.Title level={4} style={{ color: '#fff', margin: 0 }}>
            Stocks
          </Typography.Title>
        </div>
        <Menu theme="dark" mode="inline" selectedKeys={[selected]} items={items} />
      </Sider>
      <Layout>
        <Header
          style={{
            background: colorBgContainer,
            padding: '0 24px',
            display: 'flex',
            alignItems: 'center',
          }}
        >
          <Typography.Text type="secondary">
            LSTM 股价预测 · 训练 / 验证指标可追溯
          </Typography.Text>
        </Header>
        <Content style={{ margin: 24 }}>
          <div
            style={{
              padding: 24,
              minHeight: 360,
              background: colorBgContainer,
              borderRadius: borderRadiusLG,
            }}
          >
            <Outlet />
          </div>
        </Content>
      </Layout>
    </Layout>
  )
}
