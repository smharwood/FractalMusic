"""
AWS Lambda function
"""
import os, json
import base64
from chaos_music import GetParameters, GenerateImage, plotter, USE_FORTRAN

def lambda_handler(event, context):
    """
    Form of event:
    {
    "resource": "Resource path",
    "path": "Path parameter",
    "httpMethod": "Incoming request's method name"
    "headers": {String containing incoming request headers}
    "multiValueHeaders": {List of strings containing incoming request headers}
    "queryStringParameters": {query string parameters }
    "multiValueQueryStringParameters": {List of query string parameters}
    "pathParameters":  {path parameters}
    "stageVariables": {Applicable stage variables}
    "requestContext": {Request context, including authorizer-returned key-value pairs}
    "body": "A JSON string of the request payload."
    "isBase64Encoded": "A boolean flag to indicate if the applicable request payload is Base64-encoded"
    }
    """
    # TODO: make use of Matplotlib kosher (see logs)

    name = 'fractal'
    iterations = 1e5
    max_iterations = 5e4
    params = event['queryStringParameters']
    # Set max iterations higher if not a test
    try:
        req_con = event['requestContext']
        if req_con['stage'] == 'pub':
            max_iterations = 5e5
    except Exception: pass
    try:
        iterations = int(params['iterations'])
    except Exception: pass
    iterations = min(iterations, max_iterations)
    beats = [2,3,5,7,11,13]
    input_wts = []
    for i in range(6):
        try:
            input_wts.append(float(params['w{}'.format(i+1)]))
        except Exception:
            if i > 2 : break
            input_wts.append(1.0/beats[i])

    print(event)
    print("Using Fortran: {}".format(USE_FORTRAN))
    print("Raw weights: {}".format(input_wts))

    # Generate fractal
    # note that scratch space is at /tmp
    image_name = "/tmp/{}.png".format(name)
    generate_fractal(image_name, input_wts, iterations)
    with open(image_name, 'rb') as image:
        byte_string = base64.b64encode(image.read())
        return {
            'headers': { "Content-Type": "image/png",
                         "Access-Control-Allow-Origin": "*" },
            'statusCode': 200,
            'body': byte_string.decode('utf-8'),
#            'body': byte_string,
            'isBase64Encoded': True
        }
    # else
    return {
        'headers': { "Content-type": "text/html",
                     "Access-Control-Allow-Origin": "*" },
        'statusCode': 200,
        'body': "<h1>I guess this did not work</h1>",
    }


def generate_fractal(image_name, input_wts, n_iterations):  
    n_pts = len(input_wts)
    basis_pts, _, move_fracs = GetParameters(n_pts)
    density = GenerateImage(basis_pts, input_wts, move_fracs,
            n_Iterations=n_iterations, use_fortran=USE_FORTRAN)
    plotter(density, image_name, invert=True)
    return
