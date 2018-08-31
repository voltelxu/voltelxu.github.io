#### DispatcherServlet

springmvc围绕前端控制模式来设计，使用一个核心的servlet (dispathcerservlet) 来对请求进行处理，而实际的却是由可以配置的 delegate(委派)组件来执行。