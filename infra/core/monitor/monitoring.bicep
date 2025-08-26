@description('The location for the monitoring resources')
param location string = resourceGroup().location

@description('The name of the Log Analytics workspace')
param logAnalyticsName string

@description('The name of the Application Insights resource')
param applicationInsightsName string

resource logAnalyticsWorkspace 'Microsoft.OperationalInsights/workspaces@2022-10-01' = {
  name: logAnalyticsName
  location: location
  properties: {
    sku: {
      name: 'PerGB2018'
    }
    retentionInDays: 30
    features: {
      searchVersion: 1
      legacy: 0
      enableLogAccessUsingOnlyResourcePermissions: true
    }
  }
}

resource applicationInsights 'Microsoft.Insights/components@2020-02-02' = {
  name: applicationInsightsName
  location: location
  kind: 'web'
  properties: {
    Application_Type: 'web'
    WorkspaceResourceId: logAnalyticsWorkspace.id
    IngestionMode: 'LogAnalytics'
    publicNetworkAccessForIngestion: 'Enabled'
    publicNetworkAccessForQuery: 'Enabled'
  }
}

@description('The name of the Log Analytics workspace')
output logAnalyticsWorkspaceName string = logAnalyticsWorkspace.name

@description('The resource ID of the Log Analytics workspace')
output logAnalyticsWorkspaceId string = logAnalyticsWorkspace.id

@description('The name of the Application Insights resource')
output applicationInsightsName string = applicationInsights.name

@description('The connection string for Application Insights')
output applicationInsightsConnectionString string = applicationInsights.properties.ConnectionString

@description('The instrumentation key for Application Insights')
output applicationInsightsInstrumentationKey string = applicationInsights.properties.InstrumentationKey
